# clean_macro_from_csv.py
import pandas as pd
import numpy as np
from pathlib import Path
import sys

INPUT_PATH  = Path("data/MACRO.csv")          # put your csv path here
OUTPUT_PATH = Path("data_used/macro_clean.parquet")
COVERAGE_TH = 0.80                                 # keep cols with >=80% coverage
ROLL_WIN    = 3                                    # 3-month smoothing

def read_csv_safely(path: Path) -> pd.DataFrame:
    """Try UTF-8/BOM then GBK; return DataFrame."""
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return pd.read_csv(path, encoding="gbk", errors="ignore")

def find_date_col(df: pd.DataFrame) -> str:
    """Guess the date column name (case-insensitive)."""
    candidates = [c for c in df.columns
                  if str(c).strip().lower() in ("date","month","period","t","time")]
    if candidates:
        return candidates[0]
    # if first column looks like a date, use it
    c0 = df.columns[0]
    try:
        pd.to_datetime(df[c0])
        return c0
    except Exception:
        pass
    # fall back: create a date index from the first column
    return c0

def coerce_numeric_col(s: pd.Series) -> pd.Series:
    """Make a column numeric. Remove commas, percents, common tokens."""
    if s.dtype == "object":
        s = s.astype(str).str.strip()
        # common junk to NaN
        s = s.replace(
            {
                "—": np.nan, "–": np.nan, "-": np.nan,
                "N/A": np.nan, "n/a": np.nan, "NA": np.nan,
                "Source: Wind": np.nan, "": np.nan
            }
        )
        # remove percent signs/commas/spaces
        s = (s.str.replace("%", "", regex=False)
               .str.replace(",", "", regex=True)
               .str.replace("\u3000", "", regex=False)  # full-width space
               .str.replace(r"\s+", "", regex=True))
    return pd.to_numeric(s, errors="coerce")

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add a few sensible spreads if inputs are present."""
    out = df.copy()

    # Yield slope 10Y-1Y (best-effort name match)
    c10 = [c for c in out.columns if ("10Y" in c or "10 Y" in c) and "Yield" in c]
    c1  = [c for c in out.columns if ("1Y"  in c or "1 Y"  in c) and "Yield" in c]
    if c10 and c1:
        out["Yield_Slope_10Y_1Y"] = out[c10[0]] - out[c1[0]]

    # Inflation gap CPI-PPI (YoY)
    cpi = [c for c in out.columns if "CPI" in c and "YoY" in c]
    ppi = [c for c in out.columns if "PPI" in c and "YoY" in c]
    if cpi and ppi:
        out["Inflation_Gap_CPI_PPI"] = out[cpi[0]] - out[ppi[0]]

    # Money gap M2 - M1 (YoY)
    m1 = [c for c in out.columns if "M1: YoY" in c]
    m2 = [c for c in out.columns if "M2: YoY" in c]
    if m1 and m2:
        out["Money_Gap_M2_M1"] = out[m2[0]] - out[m1[0]]

    return out

def main(in_path: Path, out_path: Path):
    print(f"Loading raw macro CSV: {in_path}")
    raw = read_csv_safely(in_path)

    # pick/parse date
    date_col = find_date_col(raw)
    raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce")
    raw = raw.dropna(subset=[date_col])           # drop rows without a valid date
    raw = raw.set_index(date_col).sort_index()

    # drop duplicated columns if any
    raw = raw.loc[:, ~raw.columns.duplicated()]

    # coerce all to numeric
    df = raw.apply(coerce_numeric_col)

    # resample to month-end and forward-fill within month
    df = df.resample("ME").ffill()

    # coverage filter (after coercion/resample)
    coverage = df.notna().mean().sort_values(ascending=False)
    keep_cols = coverage[coverage >= COVERAGE_TH].index.tolist()
    print(f"Keeping {len(keep_cols)}/{df.shape[1]} variables with >= {int(COVERAGE_TH*100)}% coverage.")
    df = df[keep_cols]

    # 3-month rolling mean smoothing
    df = df.rolling(ROLL_WIN, min_periods=1).mean()

    # add derived features
    df = add_derived_features(df)

    # z-score standardize (avoid divide-by-zero)
    df = (df - df.mean()) / df.std(ddof=0)
    df = df.replace([np.inf, -np.inf], np.nan)

    # final coverage pass after transforms
    coverage2 = df.notna().mean()
    df = df.loc[:, coverage2 >= COVERAGE_TH]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    print(f"✅ Saved cleaned macro data -> {out_path}")
    print(f"Shape: {df.shape} | Range: {df.index.min().date()} → {df.index.max().date()}")

if __name__ == "__main__":
    # Optional CLI: python clean_macro_from_csv.py raw.csv out.parquet
    if len(sys.argv) >= 2:
        INPUT_PATH = Path(sys.argv[1])
    if len(sys.argv) >= 3:
        OUTPUT_PATH = Path(sys.argv[2])
    main(INPUT_PATH, OUTPUT_PATH)
