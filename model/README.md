# Model ECG (Inception1D) - pelny opis matematyczny i praktyczny

Ten dokument ma byc "wykladem od zera" dla osoby, ktora nie ufa, ze deep learning dziala.
Dlatego idziemy krok po kroku: co jest na wejsciu, co dzieje sie wewnatrz, jak liczony jest blad,
jak z bledu powstaja gradienty i jak dokladnie aktualizowane sa wagi.

## 0) Jedno zdanie na start

Model uczy sie funkcji:

- wejscie: 12-kanalowy przebieg EKG z 10 sekund,
- wyjscie: 8 liczb (po jednej na klase),
- trening: tak zmieniac wagi, aby roznica miedzy predykcja i etykieta byla jak najmniejsza.

Nie ma recznych regul eksperckich typu "if-else". Reguly sa zakodowane w wagach,
ktore model sam dopasowuje matematycznie.

## 1) Co dokladnie trafia do sieci

### 1.1 Dane sygnalu

Po przygotowaniu danych przez `data/filter_data.py` mamy rekordy WFDB (`.dat`, `.hea`) i metadata CSV.
Dataset (`model/training/dataset.py`) dla jednej probki zwraca:

- `x` o ksztalcie `(12, 5000)`
- `y` o ksztalcie `(8,)`

Interpretacja:

- `12` = 12 odprowadzen EKG,
- `5000` = 5000 probek czasu (10 sekund przy 500 Hz),
- kazda probka sygnalu to liczba rzeczywista (float), np. amplituda napiecia.

**Wazne:**

- wejscie do neuronow to nie 0/1,
- to sa ciagle wartosci pomiarowe.

### 1.2 Etykiety

W CSV etykiety klas (`class_*`) sa procentami `0..100`.
W datasetcie sa normalizowane do `0..1`:

\[
y_k = \frac{\text{class\_value}_k}{100}
\]

To sa soft-labels (np. 0.73), a nie tylko twarde 0/1.

## 2) Przeplyw tensora przez model: od `(12,5000)` do `(8,)`

Implementacja: `model/models/inception1d.py`.
W batchu, wejscie ma ksztalt `(B, 12, 5000)`.

### 2.1 Stem

`Conv1d(12 -> 32, kernel=1)`:

- wejscie: `(B, 12, 5000)`
- wyjscie: `(B, 32, 5000)`

Kernel 1x1 w czasie miesza kanaly (odprowadzenia), nie zmienia dlugosci czasu.

### 2.2 InceptionBlock (pojedynczy blok)

Kazdy blok ma 4 galezie:

1. `bottleneck 1x1` -> `conv k=9`
2. `bottleneck 1x1` -> `conv k=19`
3. `bottleneck 1x1` -> `conv k=39`
4. `maxpool k=3,s=1,p=1` -> `conv 1x1`

Kazda galaz daje `(B, 32, T)`, potem:

- konkatenacja po kanalach -> `(B, 128, T)`
- `BatchNorm1d(128)`
- `ReLU`

Czyli blok mapuje `(..., C_in, T)` na `(..., 128, T)`.

### 2.3 Cale grupy blokow

- Grupa 1: `block1, block2, block3` + skip connection z projekcja `1x1 (32->128)`
- Grupa 2: `block4, block5, block6` + skip identity (128->128)

Kszalty:

1. po stem: `(B, 32, 5000)`
2. po block1: `(B, 128, 5000)`
3. po block2: `(B, 128, 5000)`
4. po block3: `(B, 128, 5000)`
5. po dodaniu skip1 + BN + ReLU: `(B, 128, 5000)`
6. po block4/5/6: nadal `(B, 128, 5000)`
7. po dodaniu skip2 + BN + ReLU: `(B, 128, 5000)`

### 2.4 Wyjscie klasyfikatora

- global mean po osi czasu: `(B, 128, 5000) -> (B, 128)`
- dropout: `(B, 128)`
- `Linear(128, 8)` -> logity `(B, 8)`

Potem na inferencji: `sigmoid(logity)` -> prawdopodobienstwa `(B, 8)`.

## 3) Wzory matematyczne operacji w sieci

### 3.1 Konwolucja 1D

Dla kanalu wyjsciowego `o`, czasu `t`:

\[
z_{o,t} = b_o + \sum_{c=1}^{C_{in}} \sum_{u=0}^{K-1} W_{o,c,u}\,x_{c,t+u-p}
\]

- `K` = rozmiar kernela (np. 9, 19, 39),
- `p` = padding,
- `W` = wagi filtra,
- `b_o` = bias.

### 3.2 BatchNorm (tryb treningowy)

Dla kazdego kanalu:

\[
\mu = \frac{1}{m}\sum_i x_i,
\quad
\sigma^2 = \frac{1}{m}\sum_i (x_i-\mu)^2
\]
\[
\hat{x}_i = \frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}},
\quad
y_i = \gamma \hat{x}_i + \beta
\]

### 3.3 ReLU

\[
\text{ReLU}(x)=\max(0,x)
\]

### 3.4 Sigmoid na wyjsciu

\[
\sigma(z)=\frac{1}{1+e^{-z}}
\]

Daje liczby `0..1` dla kazdej klasy niezaleznie (multi-label).

## 4) Funkcja straty: jak liczony jest blad

W kodzie (`model/training/train_pipeline.py`) sa 2 opcje:

1. `FocalLoss`
2. `TolerantImbalanceBCELoss` (BCE z `pos_weight` i tolerancja 1%)

### 4.1 BCEWithLogits (baza)

Dla jednego logitu `z` i etykiety `y in [0,1]`:

\[
\ell_{bce}(z,y) = -\Big(y\log\sigma(z) + (1-y)\log(1-\sigma(z))\Big)
\]

Z `pos_weight = p` (dla klasy niezbalansowanej):

\[
\ell_{wbce}(z,y) = -\Big(p\,y\log\sigma(z) + (1-y)\log(1-\sigma(z))\Big)
\]

### 4.2 FocalLoss

W kodzie:

\[
p_t = \sigma(z)\,y + (1-\sigma(z))(1-y)
\]
\[
\ell_{focal} = \alpha_t (1-p_t)^\gamma \cdot \ell_{bce}
\]

- `gamma=2.0`,
- `alpha_t` zalezy od klasy i etykiety.

Sens: zmniejsza wplyw latwych przypadkow, wzmacnia trudne.

### 4.3 TolerantImbalanceBCELoss (zmiana 1%)

W kodzie:

1. `base_loss = BCEWithLogits(..., pos_weight=...)`
2. `abs_err = |sigmoid(z).detach() - y|`
3. `ramp = clip(abs_err / (2*tolerance), 0, 1)`
4. `scale = in_tolerance_weight + (1-in_tolerance_weight) * ramp^2`
5. `loss = mean(base_loss * scale)`

Dla obecnych ustawien:

- `tolerance = 0.01` (1 punkt procentowy),
- `in_tolerance_weight = 0.15`.

To znaczy: gdy blad jest bardzo maly, kara jest oslabiona, ale nie zerowa.

## 5) Propagacja wsteczna: skad wiadomo, ktore wagi zmienic

To jest sedno.

### 5.1 Regula lancuchowa

Niech `L` to strata, a `\theta` dowolna waga. Wtedy:

\[
\frac{\partial L}{\partial \theta}
= \frac{\partial L}{\partial z}
\cdot
\frac{\partial z}{\partial a}
\cdot
\frac{\partial a}{\partial \theta}
\]

W sieci mamy dlugi lancuch warstw, wiec pochodne sa mnozone po drodze.
To jest dokladnie to, co robi autograd w `loss.backward()`.

### 5.2 Dla wyjscia sigmoid + BCE (intuicja)

Dla klasycznego przypadku (bez dodatkowych wag):

\[
\frac{\partial L}{\partial z} = \sigma(z)-y
\]

Interpretacja:

- jesli `\sigma(z)` jest za duze wzgledem `y`, gradient dodatni -> trzeba zmniejszac logit,
- jesli za male, gradient ujemny -> trzeba zwiekszac logit,
- im wiekszy blad, tym zwykle wiekszy modul gradientu.

### 5.3 Dlaczego "wiemy", ktory perceptron ruszyc

Kazda waga ma swoj gradient `\partial L/\partial \theta`.
Jesli dana waga prawie nie wplywa na blad, gradient jest bliski 0.
Jesli mocno wplywa, gradient ma duzy modul.

Czyli decyzja nie jest reczna; wynika bezposrednio z rachunku rozniczkowego.

## 6) Liczbowy mini-przyklad backprop (jeden neuron)

Zalozmy prosty neuron:

\[
z = w x + b,
\quad
\hat{y}=\sigma(z)
\]

Przyjmijmy:

- `x = 0.8`
- `w = 0.5`
- `b = -0.1`
- etykieta `y = 1.0`

Krok 1: forward

\[
z = 0.5\cdot0.8 - 0.1 = 0.3
\]
\[
\hat{y}=\sigma(0.3) \approx 0.5744
\]

Krok 2: gradient po logitcie (BCE)

\[
\frac{\partial L}{\partial z}=\hat{y}-y=0.5744-1=-0.4256
\]

Krok 3: gradienty po parametrach

\[
\frac{\partial L}{\partial w}=\frac{\partial L}{\partial z}\cdot\frac{\partial z}{\partial w}
=(-0.4256)\cdot x
=(-0.4256)\cdot 0.8=-0.34048
\]
\[
\frac{\partial L}{\partial b}=\frac{\partial L}{\partial z}\cdot\frac{\partial z}{\partial b}
=-0.4256
\]

Krok 4: update (prosty gradient descent, `lr=0.001`)

\[
w_{new}=w-lr\cdot\frac{\partial L}{\partial w}
=0.5-0.001(-0.34048)
=0.50034048
\]
\[
b_{new}=b-lr\cdot\frac{\partial L}{\partial b}
=-0.1-0.001(-0.4256)
=-0.0995744
\]

Wagi wzrosly, bo predykcja byla za niska wzgledem `y=1`.
To samo dzieje sie dla milionow wag naraz w pelnym modelu.

## 7) Jak dziala AdamW liczbowo (idea)

W praktyce nie uzywamy "golego" gradient descent, tylko `AdamW`:

\[
m_t=\beta_1 m_{t-1} + (1-\beta_1)g_t
\]
\[
v_t=\beta_2 v_{t-1} + (1-\beta_2)g_t^2
\]
\[
\hat{m}_t = \frac{m_t}{1-\beta_1^t},
\quad
\hat{v}_t = \frac{v_t}{1-\beta_2^t}
\]
\[
\theta_t = \theta_{t-1} - lr\left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon} + \lambda\theta_{t-1}\right)
\]

- `g_t` = aktualny gradient,
- `m_t` = srednia gradientu,
- `v_t` = srednia kwadratu gradientu,
- `\lambda` = weight decay.

Sens praktyczny: stabilniejsze kroki i lepsza kontrola skali update.

## 8) Jak interpretowac wynik koncowy

Model zwraca 8 logitow, po sigmoid mamy 8 prawdopodobienstw.
To nie jest "jedna klasa na sile".

Przy progu `threshold=0.5`:

- `p_k >= 0.5` -> klasa dodatnia,
- `p_k < 0.5` -> klasa ujemna.

W `model/inference_api.py`, dla dlugich sygnalow:

1. sygnal dzielony jest na okna 10 s,
2. model liczy predykcje dla kazdego okna,
3. wynik koncowy = srednia prawdopodobienstw po oknach.

To zmniejsza ryzyko, ze pojedynczy fragment zakloci diagnoze.

## 9) Dlaczego to dziala w praktyce (bez "wiary")

1. Model to funkcja parametryczna o duzej pojemnosci (462216 parametrow dla 8 klas).
2. Definiujemy mierzalny cel (funkcja straty).
3. Rachunek rozniczkowy daje kierunek poprawy dla kazdej wagi.
4. Tysiace iteracji stopniowo zmniejszaja blad na danych treningowych.
5. Walidacja/test sprawdzaja, czy poprawa generalizuje na nowe dane.

To nie jest magia - to optymalizacja numeryczna + statystyka + duzo danych.

## 10) Szybkie komendy

Uruchamiaj z katalogu glownego projektu:

```powershell
python -m model.training.validate_pipeline
python -m model.training.train_pipeline --sanity
python -m model.test_model_viewer --split test
```

## 11) Artefakty treningu

W `model/annotations/`:

- `best_model.pt` - najlepszy checkpoint,
- `last_model.pt` - ostatni checkpoint,
- `class_names.json` - kolejnosc klas,
- `train_log.csv` - historia epok,
- `loss_curve.png` - przebieg straty,
- `eval_results.txt` - metryki per klasa.
