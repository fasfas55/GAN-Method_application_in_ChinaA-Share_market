import pandas as pd
import tushare as ts
import time
from datetime import datetime

START_DATE = "20100101"
END_DATE   = "20251001"
SLEEP_SEC  = 0.25
LIMIT      = 5000

# load code list and convert
raw_data = pd.read_parquet("data/Ashare_rawdata_Mon.parquet")
def to_ts_code(code):
    pref, num = code.split('.')
    m = {'sh': 'SH', 'sz': 'SZ', 'bj': 'BJ'}
    return f"{num}.{m[pref]}"
ts_list = [to_ts_code(c) for c in raw_data['code'].unique().tolist()]

# init tushare
ts.set_token('')
pro = ts.pro_api()

def fetch_fina_indicator_one(code, start_date=START_DATE, end_date=END_DATE):
    pages = []
    offset = 0
    while True:
        df = pro.query(
            "fina_indicator",
            ts_code=code,
            start_date=start_date,
            end_date=end_date,
            limit=LIMIT,
            offset=offset
        )
        if df is None or df.empty:
            break
        pages.append(df)
        if len(df) < LIMIT:
            break  # last page
        offset += LIMIT
        time.sleep(SLEEP_SEC)
    return pd.concat(pages, ignore_index=True) if pages else pd.DataFrame()

# main
all_data = []
start_time = time.time()
for i, code in enumerate(ts_list, 1):
    t0 = time.time()
    try:
        df = fetch_fina_indicator_one(code)
        if not df.empty:
            df["ann_date"] = pd.to_datetime(df["ann_date"], errors="coerce")
            df["end_date"]  = pd.to_datetime(df["end_date"],  errors="coerce")
            all_data.append(df)
            print(f"[{i}/{len(ts_list)}] {code}: {len(df)} rows")
        else:
            print(f"[{i}/{len(ts_list)}] {code}: empty")
    except Exception as e:
        print(f"[{i}/{len(ts_list)}] {code}: ERROR -> {e}")
    time.sleep(SLEEP_SEC)

    elapsed = time.time() - t0
    total_elapsed = (time.time() - start_time) / 60
    print(f"   ↳ finished {code} in {elapsed:.2f}s | total elapsed: {total_elapsed:.2f} min")

# save
if all_data:
    result = pd.concat(all_data, ignore_index=True)
    result = (result.sort_values(["ts_code","ann_date","end_date"])
                    .drop_duplicates(subset=["ts_code","ann_date","end_date"], keep="last"))
    result.to_parquet("data/fina_indicator_2010_2025.parquet", index=False)
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Saved {len(result)} rows to data/fina_indicator_2010_2025.parquet")
else:
    print("No data fetched.")
