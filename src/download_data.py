"""Pobiera zbiór danych OtoMoto (2023) z Kaggle i rozpakowuje go do katalogu projektu.

model/mvp_model.py oczekuje pliku otomoto_offers_eng_23-04-2023.csv katalog wyżej niż model/.

Źródło: https://www.kaggle.com/datasets/szymoncyperski/car-sales-offers-from-otomotopl-2023
Licencja: CC BY-SA 4.0
"""
from pathlib import Path
import urllib.request
import zipfile

DATASET = "szymoncyperski/car-sales-offers-from-otomotopl-2023"
URL = f"https://www.kaggle.com/api/v1/datasets/download/{DATASET}"
PROJECT_DIR = Path(__file__).resolve().parents[1]


def main() -> None:
    archive = PROJECT_DIR / "otomoto_2023.zip"
    if not archive.exists():
        print(f"Pobieranie {URL} ...")
        urllib.request.urlretrieve(URL, archive)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(PROJECT_DIR)
        for name in zf.namelist():
            print("Rozpakowano:", PROJECT_DIR / name)


if __name__ == "__main__":
    main()
