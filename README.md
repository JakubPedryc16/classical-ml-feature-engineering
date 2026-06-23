# System Klasyfikacji Gatunków Muzycznych

Projekt realizuje zadanie automatycznego rozpoznawania gatunków muzycznych przy wykorzystaniu zaawansowanej ekstrakcji cech sygnałów audio oraz algorytmów uczenia maszynowego. System opiera się na rygorystycznych zasadach separacji danych poprzez podział na grupy utworów, co gwarantuje rzetelność wyników.

## Zbiór Danych i Przygotowanie

Projekt wykorzystuje zbiór danych GTZAN, który zawiera 1000 utworów muzycznych rozdzielonych na 10 gatunków. Każdy gatunek reprezentowany jest przez 100 próbek audio trwających 30 sekund. System automatycznie dzieli te utwory na mniejsze segmenty (domyślnie 3-sekundowe), zwiększając liczebność zbioru przy zachowaniu logicznego powiązania fragmentów z piosenką matką.

Pliki audio należy umieścić w katalogu `data/genres_original/` w strukturze:

```
data/genres_original/
├── blues/
│   ├── blues.00000.wav
│   └── ...
├── classical/
└── ...
```

Katalog `data/` nie jest śledzony w repozytorium (patrz `.gitignore`).

## Warianty Cech

System generuje trzy warianty reprezentacji audio:

| Wariant | Opis | Wymiar |
|---------|------|--------|
| **V1 (Manual)** | Średnie 13 współczynników MFCC + spectral centroid + zero crossing rate | 15 |
| **V2 (PCA)** | PCA (10 składowych) na uśrednionych mel-spektrogramach | 10 |
| **V3 (Rich)** | Rozszerzony zestaw cech GTZAN: chroma, RMS, centroid, bandwidth, rolloff, ZCR, harmony, perceptr, tempo oraz średnie/wariancje 20 MFCC | 57 |

## Struktura Projektu

```
├── main.py                 # Punkt wejścia — uruchamianie eksperymentów
├── requirements.txt        # Zależności Python
├── src/
│   ├── processors.py       # AudioPipeline — ekstrakcja cech z plików WAV
│   ├── models.py           # GenreClassifier — ewaluacja modeli (CV + test)
│   ├── tuner.py            # GenreTuner — optymalizacja hiperparametrów (Optuna)
│   ├── data_io.py          # DataIO — zapis/odczyt cech i raportów
│   ├── experiments.py      # Orkiestracja: ekstrakcja, ewaluacja bazowa, tuning
│   ├── visualizer.py       # Wizualizacje przestrzeni cech
│   └── tests/
│       └── load_test.py    # Weryfikacja obecności danych GTZAN
├── data/
│   ├── genres_original/    # Surowe pliki audio (lokalnie)
│   └── processed/          # Wyekstrahowane cechy (.npy, .csv)
└── reports/
    ├── plots/              # Wykresy cech (V1, V2, V3)
    ├── base_eval/          # Wyniki ewaluacji bazowej
    └── tuning/             # Wyniki optymalizacji Optuna
```

## Instalacja

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS
pip install -r requirements.txt
```

## Uruchomienie

Wszystkie eksperymenty są sterowane z `main.py`. Odkomentuj wybrany etap:

```python
# 1. Ekstrakcja cech i wizualizacje (wymagane przy pierwszym uruchomieniu lub zmianie segment_duration)
run_extraction_and_viz(duration=3)

# 2. Ewaluacja bazowa — 4 modele × 3 warianty cech (CV vs test)
run_base_evaluation()

# 3. Optymalizacja hiperparametrów Optuna dla wybranego wariantu
run_full_tuning(variant="V3")
```

```bash
python main.py
```

### Weryfikacja danych

```bash
python src/tests/load_test.py
```

## Modele i Metodologia

**Modele klasyfikacji:** Random Forest, SVM, k-NN, HistGradientBoosting.

**Walidacja:** `GroupKFold` (5 foldów) oraz `GroupShuffleSplit` (80/20) z grupowaniem po utworach — segmenty z tej samej piosenki nie trafiają jednocześnie do zbioru treningowego i testowego.

**Optymalizacja:** `GenreTuner` wykorzystuje Optunę (domyślnie 30 prób na model) z `GroupKFold` w pipeline ze `StandardScaler`.

## Wyniki

Po uruchomieniu eksperymentów wyniki zapisywane są automatycznie:

- `data/processed/` — macierze cech (`X_v1.npy`, `X_v2.npy`, `X_v3.npy`, `y_labels.npy`) oraz pliki CSV
- `reports/plots/` — scatter ploty V1/V2, profile MFCC dla V3
- `reports/base_eval/` — macierze pomyłek, raporty tekstowe, `full_evaluation_comparison.csv`
- `reports/tuning/` — wyniki po tuningu, `tuning_comparison_{variant}.csv`
