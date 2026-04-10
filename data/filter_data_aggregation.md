# Zasady redukcji PTB-XL do 8 klas

Ten dokument opisuje aktualna logike z `data/filter_data.py`:
agregacje etykiet SCP do 8 klas, filtrowanie rekordow oraz split 70/10/20.

## Cel

- Zmapowac `scp_codes` do 8 wspieranych klas.
- Zachowac pelna informacje diagnostyczna w metadanych (`scp_codes_full`).
- Przygotowac stabilny split stratified do treningu (`primary_class_8`).

## Aktualne klasy docelowe i reguly agregacji

W kazdym rekordzie wyliczane sa kolumny `class_<nazwa_klasy>`.
Kazdy wynik jest liczony jako:

`min(100.0, suma_kodow_add + srednia_kodow_average)`

gdzie:
- `add` - kody sumowane,
- `average` - kody usredniane (tylko te obecne w `scp_codes`).

Aktualne mapowanie (`CLASS_RULES`):

1. `healthy`
   - `add`: `NORM`
   - `average`: brak

2. `front_heart_attack`
   - `add`: `INJAS`, `INJAL`
   - `average`: `AMI`, `ASMI`, `ALMI`

3. `first_degree_av_block`
   - `add`: `1AVB`
   - `average`: brak

4. `bottom_heart_attack`
   - `add`: `INJIN`
   - `average`: `IMI`, `ILMI`

5. `atrial_fibrillation`
   - `add`: `AFIB`, `AFLT`
   - `average`: brak

6. `complete_right_conduction_disorder`
   - `add`: `CRBBB`
   - `average`: brak

7. `incomplete_right_conduction_disorder`
   - `add`: `IRBBB`
   - `average`: brak

8. `complete_left_conduction_disorder`
   - `add`: `CLBBB`
   - `average`: brak

## Filtrowanie rekordow

- Dla kazdego rekordu liczony jest wektor 8 klas.
- Jesli maksymalny wynik klas (`max_score`) jest `<= 0`, rekord jest odrzucany.
- Klasa dominujaca:
  - `primary_class_8` = klasa z najwyzszym wynikiem,
  - `primary_class_probability` = jej wartosc.

## Co jest zachowywane poza 8 klasami

- Oryginalne `scp_codes` sa zachowane jako `scp_codes_full` (JSON).
- Kody spoza mapowania trafiaja do:
  - `unsupported_codes` (JSON),
  - `unsupported_total_probability` (suma ich prawdopodobienstw).

## Split 70/10/20 (stratified)

Split jest wykonywany dwuetapowo:

1. `train_test_split(..., test_size=0.30, stratify=primary_class_8)`
   - daje `train=70%` i `temp=30%`.
2. `temp` dzielone na `val` i `test`:
   - `test_size=2/3` dla `temp`,
   - finalnie: `val=10%`, `test=20%` calego zbioru.

Seed podzialu jest sterowany przez `--seed` (domyslnie `42`).

## Struktura wyjscia

Tworzone sa katalogi:

- `data/training/train/`
- `data/training/val/`
- `data/training/test/`

Kazdy split zawiera:

- pliki rekordow `.dat` i `.hea` (bez podfolderow klas),
- liste rekordow: `train_files.txt` / `val_files.txt` / `test_files.txt`,
- metadane: `train_metadata.csv` / `val_metadata.csv` / `test_metadata.csv`.

W CSV sa m.in.:

- `ecg_id`,
- `local_record_base`, `local_dat_file`, `local_hea_file`,
- dane wejscowe z `ptbxl_database.csv` (poza wycietymi kolumnami legacy),
- kolumny klas 8-klasowych i `primary_class_8`,
- `unsupported_codes`, `unsupported_total_probability`.

Usuwane sa kolumny legacy (`OLD_LABEL_COLUMNS`), m.in.:
`scp_codes`, `heart_axis`, `infarction_stadium1`, `infarction_stadium2`, `strat_fold`,
`filename_lr`, `filename_hr`, `signal_path`, `signal_file`, `wfdb_fs`, `wfdb_sig_len`, `wfdb_n_sig`.

## WFDB i walidacja rekordow

- Dla kazdego rekordu wykonywane jest `wfdb.rdheader(...)`.
- Jesli naglowek jest nieczytelny, proces przerywa blad.
- Sprawdzana jest tez obecnosc plikow `.dat` i `.hea`.

Opcja `--dry-run`:

- nie kopiuje plikow `.dat/.hea`,
- ale nadal wykonuje walidacje WFDB i generuje metadane/listy.

To pozwala zbudowac spojną, samowystarczalna strukture `data/training/*` zgodna z aktualnym pipeline.


