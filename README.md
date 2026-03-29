# GAN-Method Application in China A-Share Market

This is a replication and extension of Chen, Luyang; Pelger, Markus; Zhu, Jason (2019) — “Deep Learning in Asset Pricing.”
The project applies the Generative Adversarial Network (GAN) asset pricing framework to the Chinese A-share market (2010–2025).
The goal is to test whether deep learning–based stochastic discount factor (SDF) models remain significant in a different market environment.

## Project Overview
#### Objective: 
Rebuild and adapt the Deep Learning Asset Pricing via GAN framework for the China A-share market.
#### Market: 
A-share equities (Shanghai & Shenzhen exchanges)
#### Period Studied: 
January 2010 – September 2025 (monthly data)

### Model Framework:

| Component                        | Description                                                                |
| -------------------------------- | -------------------------------------------------------------------------- |
| **SDF Network (Generator)**      | Learns the optimal pricing kernel that prices assets under no-arbitrage.   |
| **Beta Network (Discriminator)** | Predicts expected returns and enforces cross-sectional consistency.        |
| **Training Objective**           | Min–Max adversarial loss minimizing pricing errors ( E[(M_t R_{t+1})^2] ). |

<img width="1918" height="547" alt="image" src="https://github.com/user-attachments/assets/679b6e3f-3128-4505-8033-907831b9415e" />

### Work Flow:
<img width="2043" height="868" alt="image" src="https://github.com/user-attachments/assets/5212e78a-06c4-4829-99db-0cd703285934" />


### Data Setup:
| Category              | Example Variables                             | Source                     |
| --------------------- | --------------------------------------------- | -------------------------- |
| **Past Returns**      | `r2_1`, `r12_2`, `r36_13`, `ST_Rev`, `LT_Rev` | Computed from close prices, data is from Baostock|
| **Profitability**     | `ROE`, `OP`, `PM`, `ATO`                      | Tushare fundamentals       |
| **Value**             | `BEME`, `S2P`, `CF2P`                         | Market + balance sheet     |
| **Trading Frictions** | `Beta`, `Turnover`, `IdioVol`, `Spread`       | Tushare daily data         |
| **Macro Predictors**  | `GDP`, `CPI`, `M2`, `interest rates`, `credit spreads`  | WIND             |

### Data Timeline:
| Phase      | Period    | Purpose                   |
| ---------- | --------- | ------------------------- |
| Training   | 2010–2019 | Model fitting             |
| Validation | 2020–2022 | Hyperparameter tuning     |
| Testing    | 2023–2025 | Out-of-sample performance |

### 📁 Project Structure

<pre>
GAN-Method_application_in_ChinaA-Share_market
│
├── README.md                     # Project documentation
├── .gitattributes                # Git LFS tracking rules
│
├── code/                         # All Python scripts for data, model, and preprocessing
│   ├── cleansing/                # Data cleaning scripts
│   │   ├── cleansing_macro.py
│   │   └── codename.py
│   │
│   ├── fetch/                    # Data fetching & preprocessing modules
│   │   ├── fetch_basic.py
│   │   ├── fetch_finin.py
│   │   ├── fetch_rawdata.py
│   │   └── fetch_sheets.py
│   │
│   ├── model/                    # GAN model definition & training utilities
│   │   ├── 1st & 2nd_version GAN.py
│   │   └── plot_loss.py
│   │   └── result.py
│   │
│   └── panel/                    # Panel data construction & calculations
│       ├── firm_char.py
│       └── panel_cal.py
│
├── data/                         # Input data (A-share characteristics & macro predictors)
│   ├── firm_char/                # Zipped or parquet firm characteristics data
│   │   ├── fina_indicator_2010_2025.zip
│   │   └── panel_final.zip
│   │
│   └── macro/                    # Macroeconomic predictors (to be added)
│
├── output           # Logs, GAN training history, and evaluation results
│   │
│   │──1st_version
│   │
│   │──2nd_version
│
└── result                       # Trained weights, saved models
</pre>

### Result Preview
<img width="6000" height="3600" alt="decile_panel" src="https://github.com/user-attachments/assets/86b2dfc8-2b12-4e44-ac17-3cd9ae3b517a" />
<img width="3600" height="1800" alt="firm_betas_top20" src="https://github.com/user-attachments/assets/0cf1aa4f-5163-4d68-8cd6-8f5bb90ead04" />
<img width="3600" height="1800" alt="macro_top20" src="https://github.com/user-attachments/assets/dc5c5ab0-c851-48af-968e-428d45ce8d98" />



### References
· Chen, Luyang; Pelger, Markus; Zhu, Jason (2024). Deep Learning in Asset Pricing. Management Science, Vol. 70, No. 2, pp. 714-750. doi:10.1287/mnsc.2023.4695.
