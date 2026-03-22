# Zasady redukcji PTB-XL do 8 klas

Ten dokument opisuje, jak i dlaczego etykiety SCP z PTB-XL zostaly zredukowane do 8 klas docelowych.

## Cel

- Zachowac procenty z `scp_codes` (bez utraty informacji o pozostalych chorobach).
- Utworzyc jedna etykiete dominujaca (`primary_class_8`) do stratified split 70/10/20.
- Uproscic klasy do zakresu, ktory obsluguje system.

## Klasy docelowe i operacje

W kazdym rekordzie liczone sa kolumny `class_<nazwa_klasy>`.
Kazdy wynik klasy po agregacji jest obcinany do maksymalnie `100.0`.

1. `healthy`
   - Kody: `NORM`
   - Operacja: **dodawanie** (tu praktycznie wartosc bezposrednia)
   - Uzasadnienie: to pojedynczy kod reprezentujacy brak istotnej patologii.

2. `front_heart_attack`
   - Kody usredniane: `AMI`, `ASMI`, `ALMI`
   - Kody dodawane: `INJAS`, `INJAL`
   - Operacja koncowa: **suma(dodawane) + srednia(usredniane)**, obcieta do 100
   - Uzasadnienie: kody AMI/ASMI/ALMI to blisko powiazane warianty zblizonego fenotypu, a INJAS/INJAL sa dodatkowym sygnalem urazowym dla tej samej lokalizacji.

3. `side_heart_attack`
   - Kody usredniane: `LMI`
   - Kody dodawane: `INJLA`
   - Operacja: **suma + srednia**
   - Uzasadnienie: laczenie cechy zawalu bocznego i urazu bocznego.

4. `bottom_heart_attack`
   - Kody usredniane: `IMI`, `ILMI`
   - Kody dodawane: `INJIN`
   - Operacja: **suma + srednia**
   - Uzasadnienie: analogicznie do innych lokalizacji zawalu, laczymy warianty zawalowe i urazowe.

5. `back_heart_attack`
   - Kody: `PMI`
   - Operacja: **dodawanie**
   - Uzasadnienie: pojedynczy kod reprezentujacy te klase.

6. `complete_right_conduction_disorder`
   - Kody: `CRBBB`
   - Operacja: **dodawanie**
   - Uzasadnienie: bezposrednie mapowanie 1:1.

7. `incomplete_right_conduction_disorder`
   - Kody: `IRBBB`
   - Operacja: **dodawanie**
   - Uzasadnienie: bezposrednie mapowanie 1:1.

8. `complete_left_conduction_disorder`
   - Kody: `CLBBB`
   - Operacja: **dodawanie**
   - Uzasadnienie: bezposrednie mapowanie 1:1.

## Co zostaje usuniete, a co zachowane

- **Usuwane z klasyfikacji 8-klasowej:** wszystkie kody SCP nienalezace do listy powyzej.
- **NIE gubimy informacji:** wszystkie oryginalne `scp_codes` sa zapisywane do `scp_codes_full`.
- Dodatkowo zapisywane sa:
  - `unsupported_codes` - kody spoza 8 klas,
  - `unsupported_total_probability` - suma ich procentow.

Dzieki temu model trenuje na 8 klasach, ale metadane zachowuja pelna informacje diagnostyczna.

## Wybor klasy dominujacej i split

- `primary_class_8` = klasa z najwyzszym wynikiem wsrod 8 klas.
- Split wykonujemy stratified po `primary_class_8`:
  - train: 70%
  - val: 10%
  - test: 20%

To utrzymuje podobny rozklad klas w kazdej czesci zbioru.

## Struktura wyjscia

- `data/training/train/`
- `data/training/val/`
- `data/training/test/`

Kazdy folder zawiera:
- pliki `.dat` i `.hea` (bez podfolderow klas),
- plik listy rekordow (`train_files.txt`, `val_files.txt`, `test_files.txt`),
- CSV metadanych splitu (`train_metadata.csv`, `val_metadata.csv`, `test_metadata.csv`) z:
  - wiekszoscia metadanych rekordu z `ptbxl_database.csv` (pod UI/wizualizacje),
  - informacja o pliku,
  - procentami 8 klas,
  - klasa dominujaca,
  - pelnym `scp_codes` i kodami niewspieranymi,
  - lokalnymi nazwami plikow (`local_dat_file`, `local_hea_file`).

Podczas zapisu nowego CSV usuwane sa kolumny zwiazane ze starymi oznaczeniami/podzialem,
np. `scp_codes`, `heart_axis`, `infarction_stadium1`, `infarction_stadium2`, `strat_fold`
oraz oryginalne sciezki plikow (`filename_lr`, `filename_hr`, `signal_path`, `signal_file`).

## Dlaczego dalej korzystamy z WFDB

- Dla kazdego rekordu wykonywana jest walidacja `wfdb.rdheader(...)` przed kopiowaniem plikow.
- Tylko rekordy z poprawnym naglowkiem WFDB sa przetwarzane dalej.
- Parametry techniczne z naglowka (np. czestotliwosc) nie sa zapisywane do CSV,
  bo w tym zbiorze wszystkie rekordy sa 500 Hz.

To daje pewnosc, ze po usunieciu katalogu `ptb-xl` pozostaje gotowy, sprawdzony i samowystarczalny zbior treningowy.


