import pandas as pd
import numpy as np

panel_df = pd.read_parquet("data/Ashare_rawdata_Mon.parquet")
"""check the code list"""
print((panel_df["code"].value_counts()))
print(panel_df)

panel_df['ret'] = panel_df['pctChg'] / 100
gret = panel_df.groupby('code')['ret']

# Calculate market firm predictors
def  roll_prod_minus1(series, window, shift):
    """compute the cumulative return for a simple turns"""
    s = (1+ series).shift(shift)
    # rolling np.prod on raw values, then minus 1
    rp = s.rolling(window=window, min_periods=window).apply(np.prod, raw=True)
    return rp - 1.0 # compounded return -1

def add_past_return_factors(df):
    """
    Expects columns: ['code','date','pctChg'] (monthly).
    Returns a copy with: ['ret','r2_1','r12_2','r12_7','r36_13','ST_Rev','LT_Rev'].
    """
    out = df.copy()
    out['date'] = pd.to_datetime(out['date'])
    out = out.sort_values(['code', 'date']).reset_index(drop=True)

    # monthly decimal return
    out['ret'] = out['pctChg'] / 100.0

    # group once
    g = out.groupby('code')['ret']

    # short-term momentum (t-1)
    out['r2_1'] = g.shift(1)

    # 12–2 momentum: t-12..t-2  (11 months, skip last month)
    out['r12_2'] = g.transform(lambda x: roll_prod_minus1(x, window=11, shift=2))

    # 12–7 momentum: t-12..t-7 (6 months)
    out['r12_7'] = g.transform(lambda x: roll_prod_minus1(x, window=6, shift=7))

    # 36–13 momentum: t-36..t-13 (24 months)
    out['r36_13'] = g.transform(lambda x: roll_prod_minus1(x, window=24, shift=13))

    # reversals
    out['ST_Rev'] = -out['r2_1']
    out['LT_Rev'] = -out['r36_13']

    return out

panel_df = add_past_return_factors(panel_df).dropna().reset_index(drop=True)
panel_df.to_parquet("Ashare_panel_Mon.parquet")
print(panel_df)