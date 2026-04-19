from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from model.inference_api import load_checkpoint_model
from model.training.dataset import ECGWFDBDataset
from model.training.schema import infer_file_columns, CANONICAL_CLASS_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "training"
ANNOTATIONS_DIR = PROJECT_ROOT / "model" / "annotations"

CLASS_COLORS = ['#2ca02c', '#d62728', '#ff7f0e', '#8c564b', '#9467bd', '#e377c2', '#7f7f7f', '#1f77b4']
LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


class Explainable1DCNN:
    """Hybrydowe XAI: Grad-CAM (Czas) + Input Gradients (Ważność Odprowadzeń)"""

    def __init__(self, model, target_module):
        self.model = model
        self.target_module = target_module
        self.gradients = None
        self.activations = None
        self.hook_handles = [
            target_module.register_forward_hook(self._save_act),
            target_module.register_full_backward_hook(self._save_grad)
        ]

    def _save_act(self, mod, inp, out):
        self.activations = out

    def _save_grad(self, mod, gin, gout):
        self.gradients = gout[0]

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        if input_tensor.grad is not None:
            input_tensor.grad.zero_()

        # Wymagamy gradientów na wejściu do obliczenia ważności odprowadzeń (Saliency)
        input_tensor.requires_grad_(True)

        output = self.model(input_tensor)
        output[0, class_idx].backward(retain_graph=True)

        # 1. Obliczenie Grad-CAM (Mapy w czasie)
        weights = torch.mean(self.gradients, dim=2, keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze(0)
        cam = torch.relu(cam)

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        # 2. Obliczenie Ważności Odprowadzeń (Input Saliency)
        # Sumujemy bezwzględne wartości gradientów wzdłuż osi czasu dla każdego kanału
        input_grads = input_tensor.grad[0]  # kształt (12, 5000)
        lead_importance = torch.sum(torch.abs(input_grads), dim=1)  # kształt (12,)

        # Normalizacja do % (suma = 100%)
        lead_imp_sum = torch.sum(lead_importance)
        if lead_imp_sum > 1e-8:
            lead_importance = (lead_importance / lead_imp_sum) * 100
        else:
            lead_importance = torch.zeros_like(lead_importance)

        return cam.detach().cpu().numpy(), lead_importance.detach().cpu().numpy()


def _choose_checkpoint() -> Path:
    candidates = sorted(list(ANNOTATIONS_DIR.glob("*.pt")))
    if not candidates:
        raise FileNotFoundError(f"Brak wag .pt w {ANNOTATIONS_DIR}")

    print("\n--- WYBÓR WAG MODELU ---")
    for i, c in enumerate(candidates):
        print(f"[{i}] {c.name}")

    while True:
        try:
            idx = int(input("Wybierz indeks wag: "))
            return candidates[idx]
        except (ValueError, IndexError):
            print("Niepoprawny wybór.")


def create_interactive_plot(signal, cams, importances, class_names, probs, true_labels):
    time_x = np.linspace(0, 10, 5000)

    # Konfiguracja asymetrycznej siatki Plotly
    # Lewa kolumna: 12 wierszy na sygnał EKG
    # Prawa kolumna: 1 duży wiersz (nałożony na 12) na wykres słupkowy
    specs = [[{"type": "xy"}, {"type": "bar", "rowspan": 12}]] + \
            [[{"type": "xy"}, None] for _ in range(11)]

    subplot_titles = [f"EKG {LEAD_NAMES[0]}", "Ważność Odprowadzeń (%)"] + \
                     [f"EKG {LEAD_NAMES[i]}" for i in range(1, 12)]

    fig = make_subplots(
        rows=12, cols=2,
        shared_xaxes=False,  # Nie dzielimy osi X między czasem a prawym wykresem %
        column_widths=[0.82, 0.18],
        horizontal_spacing=0.04,
        vertical_spacing=0.02,
        specs=specs,
        subplot_titles=subplot_titles
    )

    for i in range(12):
        # 1. Rysowanie czarnego sygnału EKG (lewa kolumna)
        fig.add_trace(go.Scatter(x=time_x, y=signal[i], name=f"EKG {LEAD_NAMES[i]}",
                                 line=dict(color='black', width=1), showlegend=False),
                      row=i + 1, col=1)

        # 2. Nakładanie map czasowych Grad-CAM na wykres
        for cls_idx, cam in cams.items():
            color = CLASS_COLORS[cls_idx % len(CLASS_COLORS)]
            mask = cam > 0.05

            fig.add_trace(go.Scatter(
                x=time_x[mask], y=signal[i][mask],
                mode='markers',
                marker=dict(color=color, size=4, opacity=cam[mask] * 0.6),
                name=f"{class_names[cls_idx]} (P: {probs[cls_idx]:.2f})",
                legendgroup=class_names[cls_idx],
                showlegend=(i == 0)  # Pokazujemy w legendzie tylko raz
            ), row=i + 1, col=1)

    # 3. Rysowanie wykresu słupkowego dla ważności odprowadzeń (prawa kolumna)
    for cls_idx, importance in importances.items():
        color = CLASS_COLORS[cls_idx % len(CLASS_COLORS)]
        fig.add_trace(go.Bar(
            y=LEAD_NAMES[::-1],  # Odwracamy, by 'I' było na samej górze
            x=importance[::-1],
            name=class_names[cls_idx],
            orientation='h',
            marker_color=color,
            legendgroup=class_names[cls_idx],
            showlegend=False  # Legenda obsłużona przez Scatter
        ), row=1, col=2)

    # Konfiguracja wyglądu
    fig.update_layout(
        height=1500,
        title_text="Interpretacja Diagnozy EKG (Czas + Anatomia)",
        showlegend=True,
        barmode='group',  # Słupki obok siebie, jeśli jest kilka diagnoz
        hovermode='closest'
    )

    # Dodanie opisów osi
    fig.update_xaxes(title_text="Czas [s]", row=12, col=1)
    fig.update_xaxes(title_text="Wpływ na sieć [%]", row=1, col=2)

    fig.show()


def main():
    ckpt_path = _choose_checkpoint()

    class_names = CANONICAL_CLASS_COLUMNS
    model, device = load_checkpoint_model(ckpt_path, num_classes=len(class_names), device='cpu')
    model.eval()

    split_dir = DATA_ROOT / "test"
    ds = ECGWFDBDataset(split_dir=split_dir, metadata_filename="test_metadata.csv",
                        label_columns=class_names,
                        file_columns=infer_file_columns(pd.read_csv(split_dir / "test_metadata.csv").columns))

    xai_engine = Explainable1DCNN(model, model.block6)

    while True:
        idx_raw = input("\nIndeks próbki (lub 'q'): ")
        if idx_raw == 'q': break
        idx = int(idx_raw)

        x, y = ds[idx]
        x_in = x.unsqueeze(0)

        # Ważne: włączamy gradienty, by móc liczyć Saliency
        with torch.enable_grad():
            logits = model(x_in)
            probs = torch.sigmoid(logits)[0].detach().numpy()

        active_indices = [i for i, p in enumerate(probs) if p > 0.1]

        cams = {}
        importances = {}

        print(f"\n--- WYNIKI DLA PRÓBKI {idx} ---")
        for i in active_indices:
            cam, imp = xai_engine.generate(x_in, i)
            cams[i] = cam
            importances[i] = imp

            best_lead = np.argmax(imp)
            print(
                f"[{class_names[i]}] Pewność: {probs[i]:.2f} | Główny kanał decyzyjny: {LEAD_NAMES[best_lead]} ({imp[best_lead]:.1f}%)")

        create_interactive_plot(x.numpy(), cams, importances, class_names, probs, y.numpy())


if __name__ == "__main__":
    main()