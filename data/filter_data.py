import argparse
import ast
import json
import os
import shutil
from typing import Dict, Iterable, Tuple

import pandas as pd
import wfdb
from sklearn.model_selection import train_test_split


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "ptb-xl")
PTBXL_DATABASE_PATH = os.path.join(DATASET_DIR, "ptbxl_database.csv")
TRAINING_PATH = os.path.join(BASE_DIR, "training")

# Reguly agregacji:
# - "add": sygnaly komplementarne sumujemy i obcinamy do 100.
# - "average": warianty tego samego fenotypu usredniamy.
CLASS_RULES: Dict[str, Dict[str, Iterable[str]]] = {
    "healthy": {"add": ["NORM"], "average": []},
    "front_heart_attack": {
        "add": ["INJAS", "INJAL"],
        "average": ["AMI", "ASMI", "ALMI"],
    },
    "side_heart_attack": {"add": ["INJLA"], "average": ["LMI"]},
    "bottom_heart_attack": {
        "add": ["INJIN"],
        "average": ["IMI", "ILMI"],
    },
    "back_heart_attack": {"add": ["PMI"], "average": []},
    "complete_right_conduction_disorder": {"add": ["CRBBB"], "average": []},
    "incomplete_right_conduction_disorder": {"add": ["IRBBB"], "average": []},
    "complete_left_conduction_disorder": {"add": ["CLBBB"], "average": []},
}

OLD_LABEL_COLUMNS = {
    "scp_codes",
    "heart_axis",
    "infarction_stadium1",
    "infarction_stadium2",
    "strat_fold",
    "filename_lr",
    "filename_hr",
    "signal_path",
    "signal_file",
    "wfdb_fs",
    "wfdb_sig_len",
    "wfdb_n_sig",
}

NEW_CLASS_COLUMNS = [
    "primary_class_8",
    "primary_class_probability",
    "class_healthy",
    "class_front_heart_attack",
    "class_side_heart_attack",
    "class_bottom_heart_attack",
    "class_back_heart_attack",
    "class_complete_right_conduction_disorder",
    "class_incomplete_right_conduction_disorder",
    "class_complete_left_conduction_disorder",
]


def parse_scp_codes(raw_value) -> Dict[str, float]:
    if isinstance(raw_value, dict):
        parsed = raw_value
    elif isinstance(raw_value, str):
        parsed = ast.literal_eval(raw_value)
    else:
        parsed = {}

    clean = {}
    for code, value in parsed.items():
        try:
            clean[str(code)] = float(value)
        except (TypeError, ValueError):
            continue
    return clean


def aggregate_classes(scp_codes: Dict[str, float]) -> Dict[str, float]:
    class_scores: Dict[str, float] = {}
    for class_name, rule in CLASS_RULES.items():
        add_sum = sum(scp_codes.get(code, 0.0) for code in rule["add"])
        avg_values = [scp_codes.get(code, 0.0) for code in rule["average"] if code in scp_codes]
        avg_score = (sum(avg_values) / len(avg_values)) if avg_values else 0.0
        class_scores[class_name] = min(100.0, add_sum + avg_score)
    return class_scores


def pick_signal_path(row: pd.Series) -> str:
    filename_hr = row.get("filename_hr")
    if isinstance(filename_hr, str) and filename_hr.strip():
        return filename_hr

    filename_lr = row.get("filename_lr")
    if isinstance(filename_lr, str) and filename_lr.strip():
        return filename_lr

    raise ValueError(f"Brak filename_hr/filename_lr dla ecg_id={row.name}")


def build_reduced_dataframe(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    supported_codes = {
        code
        for rule in CLASS_RULES.values()
        for code in list(rule["add"]) + list(rule["average"])
    }

    rows = []
    dropped_without_supported = 0

    for ecg_id, row in df_raw.iterrows():
        scp_codes = parse_scp_codes(row.get("scp_codes", {}))
        class_scores = aggregate_classes(scp_codes)

        max_score = max(class_scores.values()) if class_scores else 0.0
        if max_score <= 0.0:
            dropped_without_supported += 1
            continue

        primary_class = max(class_scores, key=class_scores.get)
        signal_path = pick_signal_path(row)

        unsupported_codes = {
            code: value for code, value in scp_codes.items() if code not in supported_codes
        }

        row_data = row.to_dict()
        row_data.update(
            {
                "ecg_id": ecg_id,
                "signal_path": signal_path,
                "signal_file": os.path.basename(signal_path),
                "primary_class_8": primary_class,
                "primary_class_probability": class_scores[primary_class],
                "scp_codes_full": json.dumps(scp_codes, sort_keys=True),
                "unsupported_codes": json.dumps(unsupported_codes, sort_keys=True),
                "unsupported_total_probability": sum(unsupported_codes.values()),
            }
        )

        for class_name, score in class_scores.items():
            row_data[f"class_{class_name}"] = score

        rows.append(row_data)

    return pd.DataFrame(rows), dropped_without_supported


def split_dataset(df_all: pd.DataFrame, random_state: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, temp_df = train_test_split(
        df_all,
        test_size=0.30,
        random_state=random_state,
        stratify=df_all["primary_class_8"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(2 / 3),
        random_state=random_state,
        stratify=temp_df["primary_class_8"],
    )
    return train_df, val_df, test_df


def reset_split_dir(path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def save_split(
    split_df: pd.DataFrame,
    split_name: str,
    dataset_root: str,
    output_root: str,
    dry_run: bool,
) -> None:
    split_dir = os.path.join(output_root, split_name)
    reset_split_dir(split_dir)

    copied_files = 0
    split_rows = []
    files_manifest = []
    wfdb_validated = 0

    for _, row in split_df.iterrows():
        source_base = os.path.join(dataset_root, row["signal_path"])
        local_record_base = f"{int(row['ecg_id']):05d}_{os.path.basename(row['signal_path'])}"

        try:
            header = wfdb.rdheader(source_base)
            wfdb_validated += 1
        except Exception as exc:
            raise RuntimeError(
                f"WFDB nie moze odczytac naglowka rekordu ecg_id={row['ecg_id']} ({source_base}): {exc}"
            ) from exc

        row_out = row.to_dict()
        row_out.update(
            {
                "local_record_base": local_record_base,
                "local_dat_file": local_record_base + ".dat",
                "local_hea_file": local_record_base + ".hea",
            }
        )
        split_rows.append(row_out)
        files_manifest.append(local_record_base)

        for ext in [".dat", ".hea"]:
            src = source_base + ext
            dst = os.path.join(split_dir, local_record_base + ext)

            if os.path.exists(src):
                copied_files += 1
                if not dry_run:
                    shutil.copy2(src, dst)
            else:
                raise FileNotFoundError(
                    f"Brak pliku zrodlowego dla ecg_id={row['ecg_id']}: {src}"
                )

    split_df_with_meta = pd.DataFrame(split_rows)

    base_columns = [
        "ecg_id",
        "local_record_base",
        "local_dat_file",
        "local_hea_file",
    ]
    remaining_columns = [
        col
        for col in split_df_with_meta.columns
        if col not in OLD_LABEL_COLUMNS
        and col not in base_columns
        and col not in NEW_CLASS_COLUMNS
        and col not in {"unsupported_codes", "unsupported_total_probability"}
    ]
    kept_columns = base_columns + remaining_columns + NEW_CLASS_COLUMNS + [
        "unsupported_codes",
        "unsupported_total_probability",
    ]

    split_df_clean = split_df_with_meta[kept_columns]
    csv_path = os.path.join(split_dir, f"{split_name}_metadata.csv")
    split_df_clean.to_csv(csv_path, index=False)

    file_list_path = os.path.join(split_dir, f"{split_name}_files.txt")
    with open(file_list_path, "w", encoding="utf-8") as file_list:
        file_list.write("\n".join(files_manifest) + "\n")

    print(
        f"[{split_name}] rekordow={len(split_df)}; wfdb_ok={wfdb_validated}; "
        f"skopiowane_pliki={copied_files}; csv={csv_path}; lista={file_list_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Buduje split PTB-XL 70/10/20 dla 8 klas bez podfolderow klas."
    )
    parser.add_argument("--dataset-dir", default=DATASET_DIR, help="Katalog z PTB-XL")
    parser.add_argument(
        "--database-csv",
        default=PTBXL_DATABASE_PATH,
        help="Sciezka do ptbxl_database.csv",
    )
    parser.add_argument("--output-dir", default=TRAINING_PATH, help="Katalog wyjsciowy")
    parser.add_argument("--seed", type=int, default=42, help="Seed dla splitu")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Tworzy CSV i podzial bez kopiowania plikow .dat/.hea",
    )
    args = parser.parse_args()

    df_raw = pd.read_csv(args.database_csv, index_col="ecg_id")
    df_all, dropped_without_supported = build_reduced_dataframe(df_raw)

    if df_all.empty:
        raise RuntimeError("Brak rekordow z mapowaniem do wspieranych 8 klas.")

    train_df, val_df, test_df = split_dataset(df_all, random_state=args.seed)

    print(f"Liczba rekordow PTB-XL (wejscie): {len(df_raw)}")
    print(f"Liczba rekordow po redukcji do 8 klas: {len(df_all)}")
    print(f"Usuniete bez wspieranych etykiet: {dropped_without_supported}")
    print("\nRozklad klas (calosc):")
    print(df_all["primary_class_8"].value_counts())
    print("\nRozklad klas (train/val/test):")
    print("[train]\n", train_df["primary_class_8"].value_counts())
    print("[val]\n", val_df["primary_class_8"].value_counts())
    print("[test]\n", test_df["primary_class_8"].value_counts())

    os.makedirs(args.output_dir, exist_ok=True)
    save_split(train_df, "train", args.dataset_dir, args.output_dir, dry_run=args.dry_run)
    save_split(val_df, "val", args.dataset_dir, args.output_dir, dry_run=args.dry_run)
    save_split(test_df, "test", args.dataset_dir, args.output_dir, dry_run=args.dry_run)

    print("Gotowe.")


if __name__ == "__main__":
    main()
