import pandas as pd
import numpy as np
import baostock as bs

# download list
START_DATE = '2007-01-04'
END_DATE = '2025-10-04'
STOCK_LIST_DATE_New = '2025-09-30'# For today's stock list
STOCK_LIST_DATE_OLd = '2010-01-04'# For stocks that could run for period of days
OUTPUT_DIR = 'stock_data'
ADJUST_FLAG = '2'

def get_stock_list(STOCK_LIST_DATE_New,STOCK_LIST_DATE_OLd):
    rs_today = bs.query_all_stock(day=STOCK_LIST_DATE_New)
    rs_olday = bs.query_all_stock(day=STOCK_LIST_DATE_OLd)

    dl_new = []
    dl_old = []
    while (rs_olday.error_code == '0') & rs_olday.next():
        dl_old.append(rs_olday.get_row_data())
    while (rs_today.error_code == '0') & rs_today.next():
        dl_new.append(rs_today.get_row_data())

    print(f'get raw stock list: {len(dl_old)} rows in 2010')
    print(f'get raw stock list: {len(dl_new)} rows in 2025')

    df_old = pd.DataFrame(dl_old, columns=rs_olday.fields)
    df_new = pd.DataFrame(dl_new, columns=rs_today.fields)

    if not df_old.empty:
        print(df_old['code'].str[:5].value_counts())
    if not df_new.empty:
        print(df_new['code'].str[:5].value_counts())

    df_old = df_old[
        (df_old['code'].str.startswith('sh.60')) |
        (df_old['code'].str.startswith('sh.68')) |
        (df_old['code'].str.startswith('sz.00')) |
        (df_old['code'].str.startswith('sz.30'))
        ]

    df_new = df_new[
        (df_new['code'].str.startswith('sh.60')) |
        (df_new['code'].str.startswith('sh.68')) |
        (df_new['code'].str.startswith('sz.00')) |
        (df_new['code'].str.startswith('sz.30'))
        ]

    print(f'Filtered to {len(df_old)} listed A-share stocks in 2010.')
    print(f'Filtered to {len(df_new)} listed A-share stocks in 2025.')

    # endpoint-consistent universe (present at both endpoints)
    universe_codes = sorted(set(df_old['code']) & set(df_new['code']))
    universe = pd.DataFrame({'code': universe_codes}).merge(df_new, on='code', how='left')  # using 2025 code name to name the interaction
    print(f'Filtered to {len(universe["code"])} listed A-share stocks from 2010 to 2025.')

    # # Exclude ST, *ST
    # mask_st = (
    #         df['code_name'].str.contains('ST', case=False, na=False) |
    #         df['code_name'].str.contains('退', case=False, na=False)
    # )
    # df = df[~mask_st] # if mask_st is false

    stock_list = universe['code'].to_list()
    return universe, stock_list

def download_data(code,START_DATE, END_DATE,
                   datas = 'date,open,high,low,close,volume,amount,pctChg',
                   frequency = 'm', adjustflag = '2'):
    rs = bs.query_history_k_data_plus(code,datas,
    start_date=START_DATE, end_date=END_DATE,
    frequency=frequency, adjustflag=adjustflag)

    rows = []
    while (rs.error_code == '0') & rs.next():
        rows.append(rs.get_row_data())
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=rs.fields)
    # dtypes
    num = ["open", "high", "low", "close", "volume", "amount", "pctChg"]
    df[num] = df[num].apply(pd.to_numeric, errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    return df

def build_panel(stock_list, START_DATE, END_DATE):
    data_frames = []
    for code in stock_list:
        df = download_data(code, START_DATE, END_DATE)

        if df is not None and not df.empty:
            df["code"] = code
            data_frames.append(df)

    if not data_frames:
        print("No valid data fetched.")
        return pd.DataFrame()

    panel = pd.concat(data_frames, ignore_index=True)
    panel = panel.dropna()
    n_periods = panel['date'].nunique()

    print("Number of monthly periods in total:", n_periods)
    print(panel['code'].value_counts())

    full_codes = (
        panel['code'].value_counts()
        .loc[lambda s: s == panel['date'].nunique()]
        .index
    )

    panel_full = panel[panel['code'].isin(full_codes)].copy()
    print(f"{len(full_codes)} stocks with full history across {panel['date'].nunique()} months.")

    print(f"Built panel with shape: {panel_full.shape}")
    return panel_full

if __name__ == '__main__':
    bs.login()
    df_stocks, stock_list = get_stock_list(STOCK_LIST_DATE_New,STOCK_LIST_DATE_OLd)
    panel_df = build_panel(stock_list, START_DATE, END_DATE)
    print(panel_df)

    if not panel_df.empty:
        panel_df.to_parquet("Ashare_panel_Mon.parquet", index=False)
        print("Saved to AShare_panel_Mon.parquet")
    else:
        print("No data to save.")

    bs.logout()

