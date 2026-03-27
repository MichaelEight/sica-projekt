# ECG Training Pipeline (Inception1D)

Ten moduł realizuje pełny pipeline multi-label:
 - inspekcja metadanych CSV,
 - trening Inception1D z tolerancyjną stratą opartą o `BCEWithLogits` + `pos_weight`,
 - checkpointy, log CSV, wykresy,
 - automatyczna ewaluacja na teście.

## Struktura

- `model/models/inception1d.py` - architektura Inception1D.
- `model/training/metadata_inspector.py` - analiza i wykrywanie kolumn.
- `model/training/dataset.py` - dataset WFDB i statystyki etykiet.
- `model/training/train_pipeline.py` - trening + ewaluacja końcowa.
- `model/training/evaluate.py` - metryki i raport testowy.
- `model/training/validate_pipeline.py` - szybki smoke test.

## Uruchomienie

```powershell
python -m model.training.validate_pipeline
python -m model.training.train_pipeline
python -m model.training.train_pipeline --sanity
```

Wyniki trafiają do `model/annotations/`.
