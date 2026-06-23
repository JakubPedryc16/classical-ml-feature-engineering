
from src.experiments import (
    run_extraction_and_viz, 
    run_base_evaluation, 
    run_full_tuning
)

def main():

    # Uruchom, jeśli zmieniasz duration lub dodajesz nowe pliki audio
    # run_extraction_and_viz(duration=3)

    # Wyniki zapisują się w /reports/base_eval
    # run_base_evaluation()

    # Wyniki zapisują się w /reports/tuning
    run_full_tuning(variant="V3")

if __name__ == "__main__":
    main()