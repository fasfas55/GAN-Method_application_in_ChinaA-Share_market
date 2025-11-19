import pandas as pd
import tushare as ts
import time

# login ts
pro = ts.pro_api()

pro._DataApi__token 	= ''
pro._DataApi__http_url 	= ''

start_date = '20100101'
end_date   = '20251001'

all_tc = []
limit = 3000
offset = 0

# codelist
raw_data = pd.read_parquet("../data/Ashare_rawdata_Mon.parquet")

def to_ts_code(code):
    pref, num = code.split('.')
    m = {'sh': 'SH', 'sz': 'SZ', 'bj': 'BJ'}
    return f"{num}.{m[pref]}"
ts_list = [to_ts_code(c) for c in raw_data['code'].unique().tolist()]

while True:
    tc_part = pro.trade_cal(
        exchange='SSE',
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        offset=offset
    )
    if tc_part is None or tc_part.empty:
        break

    all_tc.append(tc_part)
    print(f"Fetched {len(tc_part)} rows at offset {offset}")

    # If less than limit, you've reached the end
    if len(tc_part) < limit:
        break

    offset += limit

tc = pd.concat(all_tc, ignore_index=True)

# Keep open days
tc = tc[tc['is_open'] == 1].copy()
tc['cal_date'] = tc['cal_date'].astype(str)
tc['month'] = tc['cal_date'].str.slice(0, 6)

month_end_trade_dates = (
    tc.groupby('month')['cal_date']
      .max()
      .sort_values()
      .tolist()
)

dates = month_end_trade_dates

def fetch_daily_basic(trade_date, max_retries=3, sleep_sec=1):
    for attempt in range(1, max_retries + 1):
        try:
            df = pro.daily_basic(ts_code='', trade_date=trade_date, fields='')
            print(f"{trade_date} downloaded, {len(df)} rows")
            return df
        except Exception as e:
            print(f"[{trade_date}] attempt {attempt} failed: {e}")
            if attempt < max_retries:
                time.sleep(sleep_sec)
            else:
                return None

dfs = []
failed_dates = []

for d in dates:
    df_day = fetch_daily_basic(d)
    if df_day is None or df_day.empty:
        failed_dates.append(d)
    else:
        dfs.append(df_day)

if dfs:
    df_all = pd.concat(dfs, ignore_index=True)

    mask = df_all['ts_code'].isin(ts_list)
    cleansing_basic = df_all[mask].copy().reset_index(drop=True)
    cleansing_basic.to_parquet("cleansing_basic.parquet")

    print("Saved basic.parquet, rows:", len(df_all))
else:
    print("No data fetched.")

print("Failed dates:", failed_dates)
