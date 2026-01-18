# System Klasyfikacji Gatunków Muzycznych

Projekt realizuje zadanie automatycznego rozpoznawania gatunków muzycznych przy wykorzystaniu zaawansowanej ekstrakcji cech sygnałów audio oraz algorytmów uczenia maszynowego. System opiera się na rygorystycznych zasadach separacji danych poprzez podział na grupy utworów, co gwarantuje rzetelność wyników.

## Zbiór Danych i Przygotowanie

Projekt wykorzystuje zbiór danych GTZAN, który zawiera 1000 utworów muzycznych rozdzielonych na 10 gatunków. Każdy gatunek reprezentowany jest przez 100 próbek audio trwających 30 sekund. System automatycznie dzieli te utwory na mniejsze segmenty (domyślnie 3-sekundowe), zwiększając liczebność zbioru przy zachowaniu logicznego powiązania fragmentów z piosenką matką.
