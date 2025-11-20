import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

deciles = pd.read_csv('all_deciles.csv')
firm_imp = pd.read_csv('firm_char_importance.csv')
macro_imp = pd.read_csv('macro_importance_standardized.csv')

firm_imp = firm_imp.sort_values(ascending=False, by=['mean_beta']).reset_index(drop=True)
macro_imp = macro_imp.sort_values(ascending=False, by=['coef']).reset_index(drop=True)

# firm beta
def plot_firm_betas(firm_df, top_k=20, fname="firm_betas_plot.png"):
    df = firm_df.copy()
    df["abs_beta"] = df["mean_beta"].abs()
    df = df.sort_values("abs_beta", ascending=False).head(top_k)

    plt.figure(figsize=(12, 6))
    colors = df["mean_beta"].apply(lambda x: "tab:red" if x < 0 else "tab:blue")

    plt.bar(df["feature"], df["mean_beta"], color=colors)
    plt.xticks(rotation=90)
    plt.ylabel("Mean Cross-Sectional Beta")
    plt.title(f"Top {top_k} Characteristic Exposures of SDF")
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    print(f"[Saved] {fname}")
    plt.close()

# plot decile
def plot_top_k_deciles(all_deciles_df, firm_df, k=20, cols=5, fname="decile_panel.png"):
    # Rank by |beta| or |t_stat|
    ranked = firm_df.copy()
    ranked["abs_beta"] = ranked["mean_beta"].abs()
    top_features = ranked.sort_values("abs_beta", ascending=False).head(k)["feature"].tolist()

    rows = int(np.ceil(k / cols))

    plt.figure(figsize=(cols * 4, rows * 3))

    for idx, feat in enumerate(top_features):
        df_f = all_deciles_df[all_deciles_df["feature"] == feat].sort_values("decile")

        plt.subplot(rows, cols, idx + 1)
        plt.plot(df_f["decile"], df_f["mean_w"], marker="o")
        plt.title(feat, fontsize=10)
        plt.xticks(range(10))
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    print(f"[Saved] {fname}")
    plt.close()


# plot macro imp
def plot_macro_importance(macro_df, k=15, fname="macro_importance.png"):
    df = macro_df.copy()
    df = df[df["variable"] != "intercept"]

    df["abs_coef"] = df["coef"].abs()
    df = df.sort_values("abs_coef", ascending=False).head(k)

    plt.figure(figsize=(12,6))
    colors = df["coef"].apply(lambda x: "tab:red" if x < 0 else "tab:blue")

    plt.bar(df["variable"], df["coef"], color=colors)
    plt.xticks(rotation=90)
    plt.ylabel("Coefficient")
    plt.title(f"Top {k} Macro Factor Exposures (OLS on SDF)")
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    print(f"[Saved] {fname}")
    plt.close()

print(firm_imp)
print(macro_imp)

plot_firm_betas(firm_imp)
plot_top_k_deciles(deciles, firm_imp)
plot_macro_importance(macro_imp)