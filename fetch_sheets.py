import tushare as ts
import pandas as pd
import time

# login ts
ts.set_token('')
pro = ts.pro_api()

all_income, all_balance, all_cash = [], [], []

# Quarters 2007Q1–2025Q3
quarters = pd.period_range('2010Q1', '2025Q3', freq='Q')
for q in quarters:
    period = q.end_time.strftime('%Y%m%d')
    try:
        inc = pro.income_vip(period=period,fields='')
        bal = pro.balancesheet_vip(period=period, fields='')
        cas = pro.cashflow_vip(period=period,fields='')
        all_income.append(inc)
        all_balance.append(bal)
        all_cash.append(cas)
        print(f"Fetched {period}: income {len(inc)}, balance {len(bal)}, cash {len(cas)}")
        time.sleep(0.3)
    except Exception as e:
        print(f"Failed {period}: {e}")
        continue

pd.concat(all_income).to_parquet('data/income_2010_2025.parquet')
pd.concat(all_balance).to_parquet('data/balance_2010_2025.parquet')
pd.concat(all_cash).to_parquet('data/cashflow_2010_2025.parquet')

# cleansing data
"""convert codelist from bs to ts format"""
raw_data = pd.read_parquet("data/Ashare_rawdata_Mon.parquet")
codelist = raw_data['code'].unique().tolist()

def to_ts_code(code):
    """Convert BaoStock code like 'sh.600000' to TuShare '600000.SH'."""
    pref, num = code.split('.')
    m = {'sh': 'SH', 'sz': 'SZ', 'bj': 'BJ'}
    return f"{num}.{m[pref]}"

ts_list = [to_ts_code(c) for c in codelist]

# filter out stocks in list
def check_df(df):
    """Check the length of dataset."""
    print(f"Shape of dataset: ")
    return df.shape

income = pd.read_parquet('data/income_2010_2025.parquet')
balance = pd.read_parquet('data/balance_2010_2025.parquet')
cash = pd.read_parquet('data/cashflow_2010_2025.parquet')
print(check_df(income))
print(check_df(balance))
print(check_df(cash))

income_df = income[income['ts_code'].isin(ts_list)].reset_index(drop=True)
balance_df = balance[balance['ts_code'].isin(ts_list)].reset_index(drop=True)
cash_df = cash[cash['ts_code'].isin(ts_list)].reset_index(drop=True)

print(check_df(income_df))
print(check_df(balance_df))
print(check_df(cash_df))

income_df.to_parquet('income.parquet', index=False)
balance_df.to_parquet('balance.parquet', index=False)
cash_df.to_parquet('cash.parquet', index=False)