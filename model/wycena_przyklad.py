"""Przykład użycia zapisanego modelu: wycena pojedynczego samochodu.

Wymaga pliku results/lightgbm_model.joblib, utworzonego przez mvp_model.py.
Uruchomienie: python wycena_przyklad.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent / "results" / "lightgbm_model.joblib"
art = joblib.load(MODEL_PATH)

# Cechy liczbowe ciągłe; pozostałe cechy niekategoryczne to flagi 0/1 (historia pojazdu, wyposażenie)
CIAGLE = ["age", "mileage_km", "mileage_per_year", "engine_cm3", "power_km",
          "fuel_city_l100", "number_of_doors", "number_of_seats", "equipment_count"]


def wycen(auto: dict) -> float:
    """Wartość rynkowa jednego auta [PLN]; niepodane flagi (historia, wyposażenie) = 0."""
    X = pd.DataFrame([auto]).reindex(columns=art["columns"])
    for c, dtype in art["cat_dtypes"].items():
        v = X[c].fillna("brak")
        X[c] = v.where(v.isin(dtype.categories)).astype(dtype)  # nieznana kategoria -> brak
    flagi = [c for c in X.columns if c not in art["cat_dtypes"] and c not in CIAGLE]
    X[flagi] = X[flagi].fillna(0)
    X["mileage_per_year"] = X["mileage_km"] / X["age"].clip(lower=1)
    X["equipment_count"] = X[[c for c in flagi if c.startswith("eq_")]].sum(axis=1)
    return float(np.exp(art["model"].predict(X))[0])


if __name__ == "__main__":
    # Wartości kategorii jak w danych źródłowych (np. "Diesel", "Manual", "Station wagon");
    # w praktyce podaje się też wyposażenie, np. "eq_rear_parking_camera": 1.
    auto = {
        "vehicle_brand": "Skoda", "vehicle_model": "Octavia", "generation": "III (2013-)",
        "fuel_type": "Diesel", "transmission": "Manual", "body_type": "Station wagon",
        "age": 5, "mileage_km": 150_000, "engine_cm3": 1968, "power_km": 150,
        "seller_type": "Dealer", "no_accident": 1,
    }
    print(f"Wycena: {wycen(auto):,.0f} PLN".replace(",", " "))
