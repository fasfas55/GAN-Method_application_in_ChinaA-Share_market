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

### Data Setup:
| Category              | Example Variables                             | Source                     |
| --------------------- | --------------------------------------------- | -------------------------- |
| **Past Returns**      | `r2_1`, `r12_2`, `r36_13`, `ST_Rev`, `LT_Rev` | Computed from close prices |
| **Profitability**     | `ROE`, `OP`, `PM`, `ATO`                      | BaoStock fundamentals      |
| **Value**             | `BEME`, `S2P`, `CF2P`                         | Market + balance sheet     |
| **Trading Frictions** | `Beta`, `Turnover`, `IdioVol`, `Spread`       | BaoStock daily data        |
| **Macro Predictors**  | `GDP`, `CPI`, `M2`, `interest rates`, `credit spreads`  | WIND / PBoC datasets       |

### Data Timeline:
| Phase      | Period    | Purpose                   |
| ---------- | --------- | ------------------------- |
| Training   | 2010–2019 | Model fitting             |
| Validation | 2020–2022 | Hyperparameter tuning     |
| Testing    | 2023–2025 | Out-of-sample performance |

### Project Structure：
📂 GAN-Method_application_in_ChinaA-Share_market
│
├── 📄 README.md                  # Project documentation
├── 📄 firm_char.py               # Script to build firm characteristics dataset
├── 📄 macro_data_preprocess.py   # Script to collect and process macroeconomic predictors
├── 📄 train_SDF.py               # GAN model training for SDF
├── 📄 train_beta.py              # Beta prediction network training
├── 📄 model_GAN.ipynb            # Notebook for SDF generation & pricing results
├── 📂 config/                    # Configuration files (JSON)
├── 📂 data/                      # Input data (A-share monthly data, macro variables)
├── 📂 output/                    # Model logs, trained weights, and evaluation results
└── 📂 results/                   # Final pricing metrics and plots


### References
· Chen, Luyang; Pelger, Markus; Zhu, Jason (2024). Deep Learning in Asset Pricing. Management Science, Vol. 70, No. 2, pp. 714-750. doi:10.1287/mnsc.2023.4695.
