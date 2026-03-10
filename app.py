"""
Asystent EKG — aplikacja Streamlit do analizy sygnałów elektrokardiograficznych.
Użycie: streamlit run app.py
"""
import os
import tempfile
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import wfdb

from src.model import build_model
from src.preprocessing import TARGET_CLASSES, CLASS_NAMES_PL, normalize_signal
from src.grad_cam import GradCAM1D

# ──────────────────────────────────────────────
# Konfiguracja strony
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Asystent EKG",
    page_icon="❤️",
    layout="wide",
)

# ──────────────────────────────────────────────
# Kolory dla klas
# ──────────────────────────────────────────────
CLASS_COLORS = {
    "NORM": "#2ecc71",   # zielony
    "MI": "#e74c3c",     # czerwony
    "NST_": "#f39c12",   # pomarańczowy
    "ISC_": "#e67e22",   # ciemny pomarańczowy
    "LBBB": "#9b59b6",   # fioletowy
    "RBBB": "#8e44ad",   # ciemny fioletowy
    "LVH": "#3498db",    # niebieski
    "RVH": "#2980b9",    # ciemny niebieski
}


def confidence_color(prob):
    """Return color based on probability level."""
    if prob >= 0.7:
        return "#e74c3c"  # czerwony — wysokie ryzyko
    elif prob >= 0.3:
        return "#f39c12"  # żółty — umiarkowane
    return "#2ecc71"      # zielony — niskie


# ──────────────────────────────────────────────
# Ładowanie modelu (cache)
# ──────────────────────────────────────────────
@st.cache_resource
def load_model(model_path="models/inception1d_best.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    model = build_model(input_channels=12, num_classes=len(TARGET_CLASSES))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    mean = checkpoint["mean"]
    std = checkpoint["std"]
    return model, mean, std, device


# ──────────────────────────────────────────────
# Wizualizacja sygnału EKG
# ──────────────────────────────────────────────
def plot_ecg_signal(signal, leads, fs, heatmap=None, heatmap_class=None):
    """
    Create interactive Plotly figure with 12-lead ECG.

    Args:
        signal: (n_samples, 12)
        leads: list of lead names
        fs: sampling frequency
        heatmap: optional Grad-CAM heatmap (n_samples,)
        heatmap_class: name of class for heatmap title
    """
    n_samples, n_leads = signal.shape
    time = np.arange(n_samples) / fs

    fig = make_subplots(
        rows=n_leads, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=[f"{lead} [mV]" for lead in leads],
    )

    for i in range(n_leads):
        # ECG signal trace
        fig.add_trace(
            go.Scatter(
                x=time, y=signal[:, i],
                mode="lines",
                line=dict(color="#1a1a2e", width=1),
                name=leads[i],
                showlegend=False,
            ),
            row=i + 1, col=1,
        )

        # Grad-CAM overlay
        if heatmap is not None:
            # Find regions of high activation (> 0.4)
            threshold = 0.4
            active = heatmap > threshold

            if active.any():
                # Create filled regions
                hm_scaled = heatmap * (signal[:, i].max() - signal[:, i].min()) * 0.3
                fig.add_trace(
                    go.Scatter(
                        x=time, y=np.where(active, hm_scaled + signal[:, i].min(), np.nan),
                        mode="lines",
                        fill="tozeroy",
                        fillcolor=f"rgba(231, 76, 60, 0.15)",
                        line=dict(color="rgba(231, 76, 60, 0.4)", width=0),
                        name="Obszar uwagi" if i == 0 else "",
                        showlegend=(i == 0),
                    ),
                    row=i + 1, col=1,
                )

    fig.update_layout(
        height=max(800, n_leads * 80),
        title_text=f"Sygnał EKG — {n_leads} odprowadzeń"
                   + (f" | Grad-CAM: {heatmap_class}" if heatmap_class else ""),
        title_x=0.5,
        showlegend=True,
        template="plotly_white",
    )

    fig.update_xaxes(title_text="Czas [s]", row=n_leads, col=1)

    # Clean up subplot titles font size
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=10)

    return fig


def plot_confidence_bars(probabilities):
    """Create horizontal bar chart of class confidences."""
    sorted_items = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    classes = [CLASS_NAMES_PL[cls] for cls, _ in sorted_items]
    probs = [p for _, p in sorted_items]
    colors = [confidence_color(p) for p in probs]

    fig = go.Figure(go.Bar(
        y=classes,
        x=probs,
        orientation="h",
        marker_color=colors,
        text=[f"{p:.1%}" for p in probs],
        textposition="auto",
    ))

    fig.update_layout(
        title="Pewność klasyfikacji",
        xaxis_title="Prawdopodobieństwo",
        xaxis=dict(range=[0, 1], tickformat=".0%"),
        height=400,
        template="plotly_white",
        margin=dict(l=300),
    )

    return fig


# ──────────────────────────────────────────────
# Główna aplikacja
# ──────────────────────────────────────────────
def main():
    # Header
    st.title("❤️ Asystent EKG")
    st.markdown("**Cyfrowy asystent wspomagający analizę sygnałów elektrokardiograficznych**")

    # Sidebar
    with st.sidebar:
        st.header("📂 Wgraj dane EKG")
        st.markdown(
            "Prześlij pliki w formacie **WFDB** (`.dat` + `.hea`).\n"
            "Oba pliki muszą mieć tę samą nazwę bazową."
        )

        uploaded_files = st.file_uploader(
            "Wybierz pliki EKG",
            type=["dat", "hea"],
            accept_multiple_files=True,
            help="Wgraj plik .dat oraz odpowiadający mu plik .hea",
        )

        st.divider()

        # Model info
        model_path = st.text_input(
            "Ścieżka do modelu",
            value="models/inception1d_best.pt",
            help="Ścieżka do wytrenowanego modelu Inception1D",
        )

        use_gradcam = st.checkbox("Generuj mapę Grad-CAM", value=True,
                                  help="Podświetl fragmenty sygnału istotne dla klasyfikacji")

        analyze_btn = st.button("🔍 Analizuj sygnał", type="primary", use_container_width=True)

        st.divider()
        st.caption(
            "⚠️ **Uwaga:** Narzędzie ma charakter wspomagający "
            "i nie zastępuje diagnozy lekarskiej. Ostateczna decyzja "
            "diagnostyczna zawsze pozostaje po stronie lekarza."
        )

    # Main area
    if not uploaded_files:
        st.info("👈 Wgraj pliki EKG w panelu bocznym, aby rozpocząć analizę.")

        # Show instructions
        with st.expander("ℹ️ Jak korzystać z aplikacji?"):
            st.markdown("""
            1. **Wgraj pliki EKG** — przeciągnij lub wybierz pliki `.dat` i `.hea` w panelu bocznym.
            2. **Kliknij „Analizuj sygnał"** — model przetworzy sygnał.
            3. **Sprawdź wyniki** — zobaczysz wizualizację sygnału z podświetlonymi fragmentami
               oraz klasyfikację z poziomem pewności.

            **Format danych:** WFDB (WaveForm DataBase) — pliki `.dat` (dane) i `.hea` (nagłówek).

            **Obsługiwane klasy chorób:**
            """)
            for cls in TARGET_CLASSES:
                st.markdown(f"- {CLASS_NAMES_PL[cls]}")
        return

    # Validate uploaded files
    file_dict = {}
    for f in uploaded_files:
        name, ext = os.path.splitext(f.name)
        if name not in file_dict:
            file_dict[name] = {}
        file_dict[name][ext] = f

    # Find valid pairs
    valid_pairs = {name: files for name, files in file_dict.items()
                   if ".dat" in files and ".hea" in files}

    if not valid_pairs:
        st.error("❌ Nie znaleziono kompletnej pary plików `.dat` + `.hea`. "
                 "Upewnij się, że oba pliki mają tę samą nazwę bazową.")
        return

    # Use the first valid pair
    record_name = list(valid_pairs.keys())[0]
    pair = valid_pairs[record_name]

    st.success(f"✅ Załadowano rekord: **{record_name}**")

    # Save to temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        for ext, file_obj in pair.items():
            path = os.path.join(tmpdir, record_name + ext)
            with open(path, "wb") as f:
                f.write(file_obj.getbuffer())

        record_path = os.path.join(tmpdir, record_name)

        # Load the record
        try:
            record = wfdb.rdrecord(record_path)
            signal = record.p_signal.astype(np.float32)
            leads = record.sig_name
            fs = record.fs
        except Exception as e:
            st.error(f"❌ Błąd odczytu pliku WFDB: {e}")
            return

        # Display basic info
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Odprowadzenia", len(leads))
        col2.metric("Częstotliwość", f"{fs} Hz")
        col3.metric("Czas trwania", f"{signal.shape[0]/fs:.1f} s")
        col4.metric("Próbki", signal.shape[0])

        # Always show the signal
        st.subheader("📊 Sygnał EKG")
        fig_signal = plot_ecg_signal(signal, leads, fs)
        st.plotly_chart(fig_signal, use_container_width=True)

        if analyze_btn:
            # Check if model exists
            if not os.path.exists(model_path):
                st.error(
                    f"❌ Nie znaleziono modelu: `{model_path}`\n\n"
                    "Najpierw wytrenuj model:\n"
                    "```\npython -m src.train --data_dir data/ptb-xl-...\n```"
                )
                return

            with st.spinner("🔄 Analizuję sygnał..."):
                try:
                    model, mean, std, device = load_model(model_path)
                except Exception as e:
                    st.error(f"❌ Błąd ładowania modelu: {e}")
                    return

                # Normalize and predict
                signal_norm = normalize_signal(signal, mean, std)
                signal_norm = np.nan_to_num(signal_norm, nan=0.0)
                x = torch.tensor(signal_norm.T, dtype=torch.float32).unsqueeze(0).to(device)

                # Classification
                with torch.no_grad():
                    logits = model(x)
                    probs = torch.sigmoid(logits).cpu().numpy()[0]

                probabilities = {cls: float(probs[i]) for i, cls in enumerate(TARGET_CLASSES)}

                # Grad-CAM
                heatmap = None
                top_class = None
                if use_gradcam:
                    try:
                        # Reload model for Grad-CAM (needs fresh hooks)
                        gc_model = build_model(input_channels=12, num_classes=len(TARGET_CLASSES))
                        gc_checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                        gc_model.load_state_dict(gc_checkpoint["model_state_dict"])
                        gc_model.to(device)
                        gc_model.eval()

                        grad_cam = GradCAM1D(gc_model)
                        top_class_idx = int(np.argmax(probs))
                        top_class = TARGET_CLASSES[top_class_idx]
                        heatmap, _, _ = grad_cam.generate(
                            x, class_idx=top_class_idx, signal_length=signal.shape[0]
                        )
                    except Exception as e:
                        st.warning(f"⚠️ Grad-CAM niedostępny: {e}")

            # ── Results ──
            st.divider()
            st.subheader("🩺 Wyniki klasyfikacji")

            # Top prediction
            sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
            top_cls, top_prob = sorted_probs[0]

            col_left, col_right = st.columns([1, 2])

            with col_left:
                st.markdown("### Główna predykcja")
                color = confidence_color(top_prob)
                st.markdown(
                    f'<div style="background-color:{color}20; border-left: 5px solid {color}; '
                    f'padding: 15px; border-radius: 5px;">'
                    f'<h3 style="margin:0; color:{color};">{CLASS_NAMES_PL[top_cls]}</h3>'
                    f'<p style="margin:5px 0 0 0; font-size:1.5em; font-weight:bold;">{top_prob:.1%}</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # Show all significant predictions
                st.markdown("### Wszystkie klasy")
                for cls, prob in sorted_probs:
                    color = confidence_color(prob)
                    st.markdown(
                        f'<span style="color:{color};">●</span> '
                        f'**{CLASS_NAMES_PL[cls]}**: {prob:.1%}',
                        unsafe_allow_html=True,
                    )

            with col_right:
                fig_bars = plot_confidence_bars(probabilities)
                st.plotly_chart(fig_bars, use_container_width=True)

            # Signal with Grad-CAM overlay
            if heatmap is not None:
                st.divider()
                st.subheader("🔍 Mapa uwagi (Grad-CAM)")
                st.markdown(
                    f"Podświetlone fragmenty sygnału miały największy wpływ na predykcję klasy "
                    f"**{CLASS_NAMES_PL[top_class]}**."
                )
                fig_cam = plot_ecg_signal(
                    signal, leads, fs,
                    heatmap=heatmap,
                    heatmap_class=CLASS_NAMES_PL[top_class],
                )
                st.plotly_chart(fig_cam, use_container_width=True)

            # Disclaimer
            st.divider()
            st.warning(
                "⚠️ **Uwaga:** Wyniki mają charakter wspomagający i nie stanowią diagnozy medycznej. "
                "Ostateczna decyzja diagnostyczna zawsze pozostaje po stronie lekarza specjalisty."
            )


if __name__ == "__main__":
    main()
