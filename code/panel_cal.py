import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
import warnings

warnings.filterwarnings(
    "ignore",
    message="divide by zero encountered in log",
    module="statsmodels.regression.rolling"
)

income = pd.read_parquet("data_used/income.parquet")
cash = pd.read_parquet("data_used/cash.parquet")
balance = pd.read_parquet("data_used/balance.parquet")
finin = pd.read_parquet("data_used/fina_indicator_2010_2025.parquet")
panel = pd.read_parquet("data_used/Ashare_panel_Mon.parquet")
basic = pd.read_parquet("data_used/cleansing_basic.parquet")
index = pd.read_parquet("data_used/index.parquet")
macro = pd.read_parquet('data_used/macro_clean.parquet')

def prep_fund(df):
    df = df.copy()
    df = df.sort_values(['code', 'end_date'])
    return df

def to_ts_code(code):
    """Convert BaoStock code like 'sh.600000' to TuShare '600000.SH'."""
    pref, num = code.split('.')
    m = {'sh': 'SH', 'sz': 'SZ', 'bj': 'BJ'}
    return f"{num}.{m[pref]}"

# for income statement
income_cols = [
    'ts_code', 'end_date',
    'total_revenue', 'oper_cost', 'sell_exp', 'admin_exp',
    'fin_exp', 'n_income', 'ebit', 'compr_inc_attr_p', 'diluted_eps'
]

# for cash flow
cash_cols = [
    'ts_code', 'end_date',
    'n_cashflow_act', 'n_cashflow_inv_act',
    'n_cash_flows_fnc_act','c_pay_acq_const_fiolta'
]

# for balance sheet
balance_cols = [
    'ts_code', 'end_date',
    'total_assets', 'total_liab', 'total_hldr_eqy_inc_min_int',
    'money_cap', 'accounts_receiv', 'inventories',
    'accounts_pay', 'fix_assets_total'
]

basic_cols_needed = [
    'ts_code','trade_date','close', 'pb',
    'pe_ttm','pe','ps', 'ps_ttm',
    'dv_ttm', 'total_mv', 'circ_mv',
    'total_share', 'float_share',
]

index_cols_needed = [
    'ts_code','trade_date','pct_chg'
]

income_clean = income[income_cols]
cash_clean = cash[cash_cols]
balance_clean = balance[balance_cols]
basic_clean = basic[basic_cols_needed]
index_clean = index[index_cols_needed]

# align the names
income_clean = income_clean.rename(columns={'ts_code': 'code'})
balance_clean = balance_clean.rename(columns={'ts_code': 'code'})
cash_clean = cash_clean.rename(columns={'ts_code': 'code'})
basic_clean = basic_clean.rename(columns={'ts_code': 'code',
                                 'trade_date': 'end_date'})
index_clean = index_clean.rename(columns={'ts_code': 'code',
                                        'trade_date': 'end_date',
                                        'pct_chg': 'mkt_ret'})
# adjust to decimal:
index_clean['mkt_ret'] = index_clean['mkt_ret'] / 100.0

# align the formate in data
basic_clean['end_date'] = pd.to_datetime(basic_clean['end_date'])
index_clean['end_date'] = pd.to_datetime(index_clean['end_date'])

for df in (income_clean, cash_clean, balance_clean, basic_clean, index_clean):
    df['end_date'] = pd.to_datetime(df['end_date'])
    df.sort_values(['code', 'end_date'], inplace=True)

# merge sheets
sheet = (income_clean
    .merge(balance_clean, on=['code', 'end_date'], how='left')
    .merge(cash_clean,    on=['code', 'end_date'], how='left')
)

sheet['end_date'] = pd.to_datetime(sheet['end_date'])
sheet = sheet.sort_values(['code', 'end_date']).reset_index(drop=True)

# also for panel
panel['date'] = pd.to_datetime(panel['date'])
# transform panel code to tushare code
panel['code'] = panel['code'].astype(str).apply(to_ts_code)
panel = panel.sort_values(['code', 'date']).reset_index(drop=True)

# merge panel + sheet
sheet_for_merge = sheet.rename(columns={'end_date': 'date'})
basic_clean = basic_clean.rename(columns={'end_date': 'date'})
index_clean = index_clean.rename(columns={'end_date': 'date'})

panel_fund = (panel
            .merge(basic_clean, on=['code', 'date'], how='left')
            .merge(sheet_for_merge,on=['code', 'date'], how='left')
)

id_cols = ['code', 'date', 'ann_date', 'f_ann_date']
cols_to_ffill = [c for c in panel_fund.columns if c not in id_cols]

panel_fund = panel_fund.sort_values(['code', 'date'])

panel_fund[cols_to_ffill] = (
    panel_fund
    .groupby('code')[cols_to_ffill]
    .ffill()
)

cols = ['date', 'code'] + [c for c in panel_fund.columns if c not in ['code', 'date']]
panel_fund = panel_fund[cols].reset_index(drop=True)

# merge index
panel_fund = panel_fund.merge(index_clean[['date', 'mkt_ret']],
                                            on='date',
                                            how='left')

print(panel_fund)

def investment_factors(panel_fund):
    # Investment: (AT_t - AT_{t-1}) / AT_{t-1}
    panel_fund['AT_lag'] = panel_fund.groupby('code')['total_assets'].shift(1)
    panel_fund['Investment'] = (panel_fund['total_assets'] - panel_fund['AT_lag']) / panel_fund['AT_lag']

    # DPI2A: ΔPPE / AT
    panel_fund['PPE_lag'] = panel_fund.groupby('code')['fix_assets_total'].shift(1)
    panel_fund['DPI2A'] = (panel_fund['fix_assets_total'] - panel_fund['PPE_lag']) / panel_fund['total_assets']

    # NOA (approx, given current columns): (AT - cash) / AT
    panel_fund['NOA'] = (panel_fund['total_assets'] - panel_fund['money_cap']) / panel_fund['total_assets']

    # Net Share Issue
    panel_fund['shares_lag'] = panel_fund.groupby('code')['total_share'].shift(1)
    panel_fund['NI'] = -(np.log(panel_fund['total_share']) - np.log(panel_fund['shares_lag']))

    return panel_fund

def profitability_factors(df):
    df = df.copy()

    # SG&A
    df['XSGA'] = df[['sell_exp', 'admin_exp']].sum(axis=1)

    # OP = (SALE - COGS - SG&A) / AT
    df['OP'] = (df['total_revenue'] - df['oper_cost'] - df['XSGA']) / df['total_assets']

    # ROA = NI / AT
    df['ROA'] = df['n_income'] / df['total_assets']

    # ROE = NI / Equity
    df['ROE'] = df['n_income'] / df['total_hldr_eqy_inc_min_int']

    # ATO = Sales / avg(Assets)
    df['AT_lag_for_ATO'] = df.groupby('code')['total_assets'].shift(1)
    df['AT_avg'] = (df['total_assets'] + df['AT_lag_for_ATO']) / 2
    df['ATO'] = df['total_revenue'] / df['AT_avg']

    # Profit margin
    df['PM'] = df['n_income'] / df['total_revenue']

    # SG&A / Sales
    df['SGA2S'] = df['XSGA'] / df['total_revenue']

    # PPE / Assets
    df['D2A'] = df['fix_assets_total'] / df['total_assets']

    return df

def value_factors(df):
    df = df.copy()

    AT   = df['total_assets']
    LIAB = df['total_liab']
    CASH = df['money_cap']
    SALE = df['total_revenue']
    EQ   = df['total_hldr_eqy_inc_min_int']
    CFO  = df['n_cashflow_act']
    CAPX = df['c_pay_acq_const_fiolta']

    # ---- Define ME from total_mv (or circ_mv) ----
    df['ME'] = df['total_mv']

    # ---- A2ME: Assets to Market Equity ----
    df['A2ME'] = AT / df['ME']

    # ---- BEME: Book-to-Market ----
    BE = EQ
    df['BEME'] = BE / df['ME']   # or 1 / pb as alternative

    # ---- C: Cash to Assets ----
    df['C'] = CASH / AT

    # ---- CF: (CFO - CAPX) / BE ----
    df['CF'] = (CFO - CAPX) / BE

    # ---- CF2P: Cashflow to Price (CFO / ME) ----
    df['CF2P'] = CFO / df['ME']

    # ---- D2P: Dividend to Price ----
    df['D2P'] = df['dv_ttm']

    # ---- E2P: Earnings to Price ----
    df['E2P'] = df['diluted_eps'] / df['close_x']

    # ---- Q: approx Tobin's Q = (ME + Debt) / Assets ----
    df['Q'] = (df['ME'] + LIAB) / AT

    # ---- S2P: Sales to Price ----
    df['S2P'] = SALE / df['ME']

    # ---- Lev: Leverage = Liabilities / Assets ----
    df['Lev'] = LIAB / AT

    return df

def intangible_factors(df):
    df = df.copy()
    # Accrual
    df['AC'] = (df['n_income'] - df['n_cashflow_act']) / df['total_assets']

    # Operating leverage
    df['XSGA'] = df[['sell_exp', 'admin_exp']].sum(axis=1)
    df['OL'] = (df['oper_cost'] + df['XSGA']) / df['total_assets']

    # Price to cost margin
    df['PCM'] = (df['total_revenue'] - df['oper_cost']) / df['total_revenue']

    return df

def tradingfrictions_factors(df):
    df = df.copy()

    # Total Assets
    df['logAT'] = np.log(df['total_assets'])

    # Size
    df['LME'] = np.log(df['circ_mv'])

    # LTurnover Turnover
    df['LTN'] = np.log(df['volume'] / df['total_share'])

    # Rel2High Closeness to past year high
    df['pmax_12'] = df.groupby('code')['close_x'].rolling(12, min_periods=1).max().reset_index(0,drop=True)
    df['Rel2High'] = df['close_x'] / df['pmax_12']

    # Variance
    df['Var20'] = df.groupby('code')['ret'].rolling(20).var().reset_index(0, drop=True)

    return df

    # Beta
def add_risk_factors(df):
    df = df.sort_values(['code', 'date']).copy()
    results = []

    for code, sub in df.groupby('code'):
        sub = sub.copy()

        # If too few non-NaN observations, just fill with NaN
        if sub['ret'].notna().sum() < 12 or sub['mkt_ret'].notna().sum() < 12:
            sub['Alpha'] = np.nan
            sub['Beta'] = np.nan
            sub['IdioVol'] = np.nan
            sub['ResidVar'] = np.nan
            results.append(sub)
            continue

        y = sub['ret']
        X = sm.add_constant(sub['mkt_ret'])

        # Rolling 12-month regression; require at least 6 obs in window
        model = RollingOLS(y, X, window=12, min_nobs=6).fit()

        params = model.params
        sub['Alpha'] = params['const']
        sub['Beta'] = params['mkt_ret']

        # residuals = y - fitted
        fitted = params['const'] + params['mkt_ret'] * sub['mkt_ret']
        resid = y - fitted

        sub['IdioVol']  = resid.rolling(12, min_periods=6).std()
        sub['ResidVar'] = resid.rolling(12, min_periods=6).var()

        results.append(sub)

    return pd.concat(results).reset_index(drop=True)

def main(panel_fund):
    df = panel_fund.copy()
    df = investment_factors(df)
    df = profitability_factors(df)
    df = value_factors(df)
    df = intangible_factors(df)
    df = tradingfrictions_factors(df)
    df = add_risk_factors(df)
    return df

panel_factors = main(panel_fund)

# merge macro
macro = macro.reset_index().rename(columns={"Name": "date"})
macro["date"] = pd.to_datetime(macro["date"])

panel_factors = panel_factors.sort_values("date")
macro = macro.sort_values("date")

panel_factors = pd.merge_asof(
    panel_factors,
    macro,
    on="date",
    direction="backward"
)

panel_factors.drop(columns=['open','high','low', 'close_x', 'volume', 'amount', 'pctChg'], inplace=True)
print(panel_factors)

panel_factors.to_parquet('panel_final.parquet')
