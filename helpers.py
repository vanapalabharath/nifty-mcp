"""Helper functions: data loading, API calls, and calculations."""

import json
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
from scipy.optimize import brentq

# ─── Constants ───────────────────────────────────────────────────────────────

DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

API_URL = "https://www.niftyindices.com/Backpage.aspx/getHistoricaldataDBtoString"
API_HEADERS = {
    "Content-Type": "application/json; charset=utf-8",
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.niftyindices.com/Backpage.aspx",
    "X-Requested-With": "XMLHttpRequest",
    "Accept": "application/json, text/javascript, */*; q=0.01",
}

# ─── Data Loading ────────────────────────────────────────────────────────────

_INDEX_CACHE: dict | None = None


def get_indices() -> dict:
    """Get cached index data. Loads from disk on first call."""
    global _INDEX_CACHE
    if _INDEX_CACHE is None:
        _INDEX_CACHE = _load_all_index_data()
    return _INDEX_CACHE


def _load_all_index_data() -> dict:
    """Load cached index data from all_index_data.json."""
    path = os.path.join(DATA_DIR, "all_index_data.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        raw = json.load(f)
    indices = {}
    for entry in raw:
        name = entry["INDEX_NAME"]
        if not entry.get("DATA"):
            continue
        df = pd.DataFrame(entry["DATA"])
        df["date"] = pd.to_datetime(df["date"], format="%d-%b-%Y")
        df = df.sort_values("date").reset_index(drop=True)
        df["prev_close"] = df["close"].shift(1)
        df["daily_return"] = ((df["close"] - df["prev_close"]) / df["prev_close"]) * 100
        df = df.dropna(subset=["daily_return"])
        indices[name] = {
            "df": df,
            "type": entry.get("INDEX_TYPE", ""),
            "first": entry.get("FIRST_DATE", ""),
            "last": entry.get("LAST_DATE", ""),
            "points": entry.get("TOTAL_DATA_POINTS", 0),
        }
    return indices


def load_summary() -> list[dict]:
    """Load all_index_summary.json."""
    path = os.path.join(DATA_DIR, "all_index_summary.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def load_top_movers() -> dict:
    """Load top_movers.json."""
    path = os.path.join(DATA_DIR, "top_movers.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


# ─── API Helpers ─────────────────────────────────────────────────────────────


def date_to_str(dt: datetime) -> str:
    """Convert datetime to DD-Mon-YYYY string."""
    return dt.strftime("%d-%b-%Y")


def parse_api_data(d_value: str) -> list:
    """Parse the raw API response string into a list of records."""
    if not d_value or d_value.strip().startswith("Error"):
        return []
    if "\n" in d_value:
        json_array_str = d_value.split("\n", 1)[1]
        try:
            return json.loads(json_array_str)
        except Exception:
            return []
    return []


def fetch_range(index_name: str, start_date_str: str, end_date_str: str) -> list:
    """Fetch historical data from Nifty API for a date range."""
    payload = {
        "cinfo": json.dumps({
            "name": index_name,
            "startDate": start_date_str,
            "endDate": end_date_str,
            "historicaltype": "1",
            "DataType": "HR",
        })
    }
    try:
        resp = requests.post(API_URL, headers=API_HEADERS, json=payload, timeout=20)
        resp.raise_for_status()
        result = resp.json()
        return parse_api_data(result.get("d"))
    except requests.exceptions.RequestException:
        return []


# ─── Calculation Helpers ─────────────────────────────────────────────────────


def calc_xirr(dates_arr: np.ndarray, amounts_arr: np.ndarray) -> float:
    """Compute XIRR using numpy vectorized NPV + Brent's method. Returns percentage."""
    if len(amounts_arr) < 2:
        return 0.0
    d0 = dates_arr.min()
    day_fracs = (dates_arr - d0).astype("timedelta64[D]").astype(np.float64) / 365.25

    def npv(rate):
        return np.sum(amounts_arr / (1 + rate) ** day_fracs)

    try:
        return brentq(npv, -0.5, 10.0, maxiter=500) * 100
    except (ValueError, RuntimeError):
        return 0.0


def compute_cagr(start_price: float, end_price: float, years: float) -> float | None:
    """Compute CAGR as a percentage."""
    if start_price <= 0 or end_price <= 0 or years <= 0:
        return None
    return ((end_price / start_price) ** (1.0 / years) - 1.0) * 100
