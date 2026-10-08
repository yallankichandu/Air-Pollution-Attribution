"""
Synthetic Benchmark Dataset Generator for Delhi NCR
====================================================
Generates a full 1-year (8,760 records) physically realistic multi-pollutant,
meteorological, and satellite fire dataset modeled on actual Delhi-NCR CPCB,
IMD, and NASA FIRMS seasonal and diurnal dynamics.
"""

import os
import sys

# Ensure parent directory is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from core.pipeline import calculate_cpcb_aqi


def generate_delhi_benchmark_dataset(
    year: int = 2024,
    seed: int = 42,
    output_path: str = None
) -> pd.DataFrame:
    """
    Synthesizes a realistic 1-year hourly air quality dataset for Delhi NCR.
    """
    np.random.seed(seed)
    timestamps = pd.date_range(start=f"{year}-01-01 00:00", end=f"{year}-12-31 23:00", freq="h")
    n = len(timestamps)

    hours = timestamps.hour.values
    day_of_year = timestamps.dayofyear.values
    day_of_week = timestamps.dayofweek.values
    is_weekend = (day_of_week >= 5).astype(float)
    month = timestamps.month.values
    day = timestamps.day.values

    # 1. Meteorology: Temperature (°C)
    # Annual cycle: min in Jan (~12°C), max in June (~38°C)
    temp_annual = 25.0 - 12.0 * np.cos(2 * np.pi * (day_of_year - 15) / 365)
    # Diurnal cycle: min at 05:00, max at 15:00 (+/- 6°C)
    temp_diurnal = 6.0 * np.sin(2 * np.pi * (hours - 8) / 24)
    temperature = np.clip(temp_annual + temp_diurnal + np.random.normal(0, 1.5, n), 3.0, 47.0)

    # 2. Meteorology: Relative Humidity (%)
    # Inversely correlated with temperature, peaks in monsoon (July-Aug) and winter fog (Dec-Jan)
    rh_annual = 50.0 + 25.0 * np.sin(2 * np.pi * (day_of_year - 180) / 365)
    # Winter fog spike in Dec-Jan
    rh_annual[np.isin(month, [12, 1])] += 15.0
    rh_diurnal = -12.0 * np.sin(2 * np.pi * (hours - 8) / 24)
    humidity = np.clip(rh_annual + rh_diurnal + np.random.normal(0, 4.0, n), 15.0, 98.0)

    # 3. Meteorology: Wind Speed (m/s)
    # Calm winds in winter (1.0 - 2.5 m/s), gusty in summer & monsoon (3.5 - 6.5 m/s)
    wind_annual = 2.8 + 1.2 * np.sin(2 * np.pi * (day_of_year - 100) / 365)
    wind_diurnal = 0.8 * np.sin(2 * np.pi * (hours - 10) / 24)
    wind_speed = np.clip(wind_annual + wind_diurnal + np.random.exponential(0.6, n), 0.5, 9.5)

    # 4. Planetary Boundary Layer / Stagnation Factor
    # Shallow boundary layer in cold winter nights (200m), deep convective mixing in hot summer afternoons (2200m)
    pbl_height = np.maximum(200.0, 40.0 * temperature + 150.0 * wind_speed - 2.5 * humidity)
    stagnation_factor = np.clip(1200.0 / pbl_height, 0.4, 3.5)

    # 5. NASA FIRMS Satellite Active Fire Counts (Punjab & Haryana Stubble Burning)
    # Peak occurs Oct 15 - Nov 25
    fire_count = np.random.poisson(15, n).astype(float)
    stubble_mask = (month == 10) & (day >= 12)
    stubble_peak = (month == 10) & (day >= 22) | ((month == 11) & (day <= 18))
    stubble_tail = (month == 11) & (day > 18) & (day <= 28)

    fire_count[stubble_mask] += np.random.normal(450, 80, np.sum(stubble_mask))
    fire_count[stubble_peak] += np.random.normal(2400, 450, np.sum(stubble_peak))
    fire_count[stubble_tail] += np.random.normal(600, 120, np.sum(stubble_tail))
    fire_count = np.maximum(0.0, fire_count)

    # 6. Source Emission Contributions
    # (A) Vehicular Traffic:
    # Rush hours 08-10 and 17-20, lower on weekends
    traffic_profile = np.full(n, 0.25)
    traffic_profile[np.isin(hours, [8, 9, 10])] = 0.90
    traffic_profile[hours == 9] = 1.0
    traffic_profile[np.isin(hours, [17, 18, 19, 20])] = 0.95
    traffic_profile[np.isin(hours, [18, 19])] = 1.05
    traffic_profile[np.isin(hours, [11, 12, 13, 14, 15, 16])] = 0.50
    traffic_profile *= np.where(is_weekend == 1, 0.70, 1.0)
    traffic_emission = traffic_profile * np.random.uniform(0.9, 1.1, n)

    # (B) Industrial Activity:
    # Daytime working hours Mon-Sat, lower on Sundays
    industry_profile = np.full(n, 0.35)
    work_mask = (hours >= 8) & (hours <= 18) & (day_of_week < 5)
    sat_mask = (hours >= 8) & (hours <= 14) & (day_of_week == 5)
    industry_profile[work_mask] = 1.0
    industry_profile[sat_mask] = 0.75
    industry_emission = industry_profile * np.random.uniform(0.85, 1.15, n)

    # (C) Stubble Smoke Inflow (Transport with 24-36h lag and NW wind advection):
    # Shift fire count by 24h
    lagged_fires = pd.Series(fire_count).shift(28).bfill().values
    stubble_smoke = (lagged_fires / 2500.0) * np.random.uniform(0.8, 1.2, n)

    # (D) Construction and Road Dust:
    # Elevated during dry windy daytime summer months
    dust_profile = (1.0 / np.maximum(humidity / 50.0, 0.4)) * (wind_speed / 3.0) * (np.isin(hours, range(8, 19)).astype(float) * 0.7 + 0.3)
    dust_emission = np.clip(dust_profile, 0.2, 3.0)

    # (E) Episodic Diwali Fireworks Surge (Nov 1 evening in 2024):
    diwali_spike = np.zeros(n)
    diwali_mask = (month == 11) & (day == 1) & (hours >= 19)
    diwali_spike[diwali_mask] = 3.5
    diwali_post = (month == 11) & (day == 2) & (hours <= 6)
    diwali_spike[diwali_post] = 2.0

    # (F) Monsoon Washout Effect (Precipitation scavenging in July-Aug):
    monsoon_scavenging = np.ones(n)
    monsoon_mask = np.isin(month, [7, 8])
    monsoon_scavenging[monsoon_mask] = 0.35

    # 7. Synthesize Pollutant Concentrations based on Physical Balances
    # Base PM2.5 (ug/m3)
    pm25_raw = (
        35.0  # Regional background
        + 42.0 * traffic_emission
        + 28.0 * industry_emission
        + 120.0 * stubble_smoke
        + 18.0 * dust_emission
        + 90.0 * diwali_spike
    ) * stagnation_factor * monsoon_scavenging + np.random.normal(0, 8.0, n)
    pm25 = np.clip(pm25_raw, 10.0, 650.0)

    # PM10 (ug/m3) - higher dust fraction
    pm10_raw = (
        pm25 * 1.35
        + 55.0 * dust_emission * (1.5 if np.isin(month, [4, 5, 6]).any() else 1.0)
        + 20.0 * industry_emission
    ) + np.random.normal(0, 12.0, n)
    pm10 = np.clip(pm10_raw, pm25 * 1.1, 950.0)

    # NO2 (ug/m3) - primary tracer of vehicular exhaust and power combustion
    no2_raw = (
        18.0
        + 45.0 * traffic_emission
        + 22.0 * industry_emission
    ) * stagnation_factor * monsoon_scavenging + np.random.normal(0, 4.0, n)
    no2 = np.clip(no2_raw, 5.0, 195.0)

    # SO2 (ug/m3) - tracer of industrial coal/oil boilers
    so2_raw = (
        8.0
        + 32.0 * industry_emission
        + 6.0 * traffic_emission
    ) * stagnation_factor * monsoon_scavenging + np.random.normal(0, 2.5, n)
    so2 = np.clip(so2_raw, 3.0, 95.0)

    # CO (mg/m3) - incomplete combustion from vehicles and biomass
    co_raw = (
        0.4
        + 1.8 * traffic_emission
        + 1.2 * stubble_smoke
        + 0.5 * industry_emission
    ) * stagnation_factor * monsoon_scavenging + np.random.normal(0, 0.15, n)
    co = np.clip(co_raw, 0.1, 7.5)

    # Assemble raw DataFrame
    df = pd.DataFrame({
        "timestamp": timestamps,
        "PM2.5": np.round(pm25, 1),
        "PM10": np.round(pm10, 1),
        "NO2": np.round(no2, 1),
        "SO2": np.round(so2, 1),
        "CO": np.round(co, 2),
        "temperature": np.round(temperature, 1),
        "humidity": np.round(humidity, 1),
        "wind_speed": np.round(wind_speed, 2),
        "fire_count": np.round(fire_count, 0).astype(int)
    }).set_index("timestamp")

    # Compute official CPCB National Air Quality Index
    df["AQI"] = np.round(df.apply(calculate_cpcb_aqi, axis=1), 0)

    if output_path:
        df.to_csv(output_path)

    return df


if __name__ == "__main__":
    out_file = "d:/eeswar/pollution_attribution/data/delhi_ncr_benchmark_hourly.csv"
    print(f"Generating benchmark dataset to {out_file}...")
    benchmark_df = generate_delhi_benchmark_dataset(year=2024, output_path=out_file)
    print(f"Generated {len(benchmark_df)} records. Sample head:")
    print(benchmark_df.head())
