# AUC i F1-score - co oznaczaja i jak je liczymy

Ten dokument wyjasnia metryki raportowane przez `model/evaluate_test_metrics.py`.

## 1) Kontekst w tym projekcie

Model jest **multi-label** (8 klas), wiec dla kazdej klasy mamy osobna decyzje:
- predykcja prawdopodobienstwa `p` z zakresu `[0, 1]`,
- etykieta binarna (po binarizacji soft-label):
  - `1`, gdy etykieta > 0,
  - `0` w przeciwnym razie.

Skrypt liczy metryki:
- **per klasa**,
- **MACRO** (srednia po klasach),
- **MICRO** (globalnie po wszystkich decyzjach klasa-probka).

## 2) AUC (ROC-AUC)

### Co sprawdza
AUC mierzy, jak dobrze model **rozroznia** probki pozytywne od negatywnych bez wybierania jednego progu.

Interpretacja:
- `1.0` - idealne rozroznianie,
- `0.5` - jak losowanie,
- `<0.5` - zwykle sygnal, ze ranking jest odwrocony.

### Jak sie liczy (intuicyjnie)
AUC to pole pod krzywa ROC, gdzie:
- os X: `FPR = FP / (FP + TN)`
- os Y: `TPR = TP / (TP + FN)`

Krzywa ROC powstaje przez przesuwanie progu od 1 do 0. AUC to calkowite pole pod ta krzywa.

W praktyce (dla jednej klasy):
- bierzemy `y_true` (0/1) i `y_prob` (prawdopodobienstwa),
- liczymy `roc_auc_score(y_true, y_prob)`.

## 3) F1-score

### Co sprawdza
F1 laczy **precyzje** i **czulosc** w jedna liczbe dla konkretnego progu (u nas domyslnie `0.5`).

Definicje:
- `Precision = TP / (TP + FP)`
- `Recall = TP / (TP + FN)`

Wzor:

`F1 = 2 * (Precision * Recall) / (Precision + Recall)`

Interpretacja:
- wysoki F1 oznacza, ze model znajduje pozytywne przypadki i nie produkuje zbyt wielu falszywych alarmow,
- niski F1 zwykle oznacza problem z precision, recall albo z oboma.

## 4) Różnica: AUC vs F1

- **AUC** ocenia jakosc rankingu prawdopodobienstw (nie wymaga jednego progu).
- **F1** ocenia jakosc decyzji binarnej po ustawieniu progu.

Dlatego mozliwe jest:
- wysokie AUC i nizsze F1 (dobry ranking, ale prog niedobrany),
- srednie AUC i niezle F1 dla konkretnego progu.

## 5) MACRO i MICRO

### MACRO
- liczysz metryke osobno dla kazdej klasy,
- robisz srednia arytmetyczna po klasach,
- kazda klasa ma taka sama wage.

### MICRO
- zliczasz globalnie TP/FP/FN (lub splaszczasz wszystkie klasy i probki),
- metryka bardziej zalezy od czestszych wzorcow w danych.

## 6) Ograniczenia i uwagi

- AUC dla klasy moze byc `nan`, gdy w secie testowym klasa ma tylko jedna wartosc (`same 0` albo `same 1`).
- F1 zalezy od progu; przy innym progu wynik moze sie istotnie zmienic.
- W multi-label warto patrzec jednoczesnie na:
  - per-klasa,
  - MACRO,
  - MICRO.

## 7) Co generuje skrypt

`model/evaluate_test_metrics.py` zapisuje:
- raport tekstowy: `model/annotations/test_metrics_sota.txt`,
- raport JSON: `model/annotations/test_metrics_sota.json`,
- histogramy:
  - `model/annotations/plots/test_metrics_sota_auc_hist.png`,
  - `model/annotations/plots/test_metrics_sota_f1_hist.png`,
  - `model/annotations/plots/test_metrics_sota_prob_hist.png`.

