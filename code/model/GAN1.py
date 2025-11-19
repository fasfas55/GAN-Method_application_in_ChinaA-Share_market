import pandas as pd
import numpy as np
import random

import torch
import torch.nn as nn


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(10086)

final_panel = pd.read_parquet('data_used/panel_final.parquet')
macro = pd.read_parquet('data_used/macro_clean.parquet')

final_panel['date'] = pd.to_datetime(final_panel['date'])
final_panel = final_panel.sort_values(['date', 'code'])

# to check the time stamp
final_panel['date'] = final_panel['date'].astype(str)
final_panel['month'] = final_panel['date'].str.slice(0, 7)

month_end_trade_dates = (
    final_panel.groupby('month')['date']
      .max()
      .sort_values()
      .tolist()
)
dates = month_end_trade_dates

# slice time stamp
final_panel['date'] = pd.to_datetime(final_panel['date'])
train_end = pd.Timestamp('2018-12-28') # 9 full years
valid_end = pd.Timestamp('2021-12-31')

# 2. Firm panel (X, R, mask)
date = final_panel['date'].drop_duplicates().sort_values()
date_index = {d: i for i, d in enumerate(date)}

code = final_panel['code'].drop_duplicates().sort_values()
code_index = {c: i for i, c in enumerate(code)}

T = len(date) # unique dates
N = len(code) # unique stocks

R = np.full((T, N), np.nan, dtype='float32')
mask = np.zeros((T, N), dtype=bool)

firm_cols = [ 'r2_1', 'r12_2', 'r12_7', 'r36_13', 'ST_Rev', 'LT_Rev',
              'Investment', 'NOA', 'DPI2A', 'NI',
              'OP', 'ROA', 'ROE', 'ATO', 'D2A','PM',
              'ME', 'A2ME', 'BEME', 'C', 'CF', 'CF2P', 'D2P', 'E2P', 'Q', 'S2P', 'Lev',
              'AC', 'OL', 'PCM',
              'logAT', 'LME', 'LTN', 'Rel2High', 'Var20', 'Alpha', 'Beta', 'IdioVol', 'ResidVar']
F = len(firm_cols)
X = np.zeros((T, N, F), dtype='float32')

for row in final_panel.itertuples():
    t = date_index[row.date]
    n = code_index[row.code]
    R[t, n] = row.ret
    mask[t, n] = True
    X[t, n, :] = [getattr(row, col) for col in firm_cols]

R = np.clip(R, -0.5, 0.5)  # cap monthly returns at [-50%, +50%]

# Normalize firm features
X_cs = X.copy()              # [T, N, F]
mask_float = mask.astype(bool)

# 1) set missing stocks to NaN so they don't affect mean/std
X_cs[~mask_float] = np.nan

# 2) cross-sectional mean/std per time and feature
means = np.nanmean(X_cs, axis=1, keepdims=True)   # [T, 1, F]
stds  = np.nanstd(X_cs, axis=1, keepdims=True)    # [T, 1, F]

# 3) For any time-feature with all NaN, set std=1, mean=0
bad = ~np.isfinite(means) | (stds < 1e-6)
means[~np.isfinite(means)] = 0.0
stds[bad] = 1.0

# 4) normalize
X_norm = (X_cs - means) / stds
X_norm = np.nan_to_num(X_norm, nan=0.0, posinf=0.0, neginf=0.0)

X = X_norm # use this X going forward

# filter macro_arr
required_cols = ['date', 'code', 'ret'] + firm_cols
all_cols = final_panel.columns.tolist()
macro_cols = macro.columns.tolist()

macro_df = (final_panel
            .drop_duplicates(subset=['date'])[ ['date'] + macro_cols ]
            .sort_values('date'))

macro_arr = macro_df[macro_cols].to_numpy(dtype='float32')

# Normalize macro
macro_means = np.nanmean(macro_arr, axis=0, keepdims=True)
macro_stds  = np.nanstd(macro_arr, axis=0, keepdims=True)
macro_stds[macro_stds < 1e-6] = 1.0

macro_arr_norm = (macro_arr - macro_means) / macro_stds
macro_arr_norm = np.nan_to_num(macro_arr_norm, nan=0.0, posinf=0.0, neginf=0.0)

macro_arr = macro_arr_norm

class SDFGenerator(nn.Module):
    def __init__(self,
                 ind_feature_dim: int,
                 macro_feature_dim: int,
                 hidden_dims=(64, 64),
                 use_rnn=False,
                 rnn_hidden=32,
                 rnn_layers=1,
                 normalize_w=True):
        super().__init__()
        self.ind_feature_dim = ind_feature_dim
        self.macro_feature_dim = macro_feature_dim
        self.use_rnn = use_rnn
        self.normalize_w = normalize_w

        if use_rnn and macro_feature_dim > 0:
            self.rnn = nn.GRU(
                input_size=macro_feature_dim,
                hidden_size=rnn_hidden,
                num_layers=rnn_layers,
                batch_first=True
            )
            macro_out_dim = rnn_hidden
        else:
            self.rnn = None
            macro_out_dim = macro_feature_dim

        layers = []
        in_dim = ind_feature_dim + macro_out_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        self.mlp = nn.Sequential(*layers)

        self.last = nn.Linear(in_dim, 1)  # weight per stock-month
        nn.init.xavier_uniform_(self.last.weight)

    def forward(self, R, X, macro, mask):
        """
        R:     [T, N]
        X:     [T, N, F]
        macro: [T, M] or None
        mask:  [T, N] bool
        """
        device = R.device
        T, N = R.shape

        # --- macro features over time ---
        if self.rnn is not None and macro is not None:
            macro_in = macro.unsqueeze(0)        # [1, T, M]
            macro_out, _ = self.rnn(macro_in)    # [1, T, H]
            macro_feat = macro_out.squeeze(0)    # [T, H]
        else:
            macro_feat = macro if macro is not None else torch.zeros(T, 0, device=device)

        # tile macro over stocks
        if macro_feat.numel() > 0:
            macro_tiled = macro_feat.unsqueeze(1).expand(T, N, macro_feat.shape[-1])
            feat = torch.cat([X, macro_tiled], dim=2)    # [T, N, F+M]
        else:
            feat = X                                     # [T, N, F]

        # mask and flatten
        mask_f = mask.bool()
        feat_masked = feat[mask_f]                       # [num_obs, Ftot]
        R_masked = R[mask_f]                             # [num_obs]

        # MLP to get weights
        h = self.mlp(feat_masked)                        # [num_obs, hidden]
        w = self.last(h).squeeze(-1)                     # [num_obs]

        # scatter back to [T, N]
        w_full = torch.zeros_like(R, device=device)
        w_full[mask_f] = w

        # SDF_t = 1 + sum_i w_{t,i} * R_{t,i}  (optionally normalized)
        weighted_R = w_full * R * mask.float()           # [T, N]
        N_i = mask.sum(dim=1).float()                    # [T]

        if self.normalize_w:
            N_bar = N_i.mean()
            sdf_core = weighted_R.sum(dim=1) / (N_i + 1e-8) * N_bar
        else:
            sdf_core = weighted_R.sum(dim=1)

        SDF = 1.0 + sdf_core                             # [T]

        return SDF, w_full

class MomentNet(nn.Module):
    def __init__(self,
                 ind_feature_dim: int,
                 macro_feature_dim: int,
                 num_moments: int = 50,
                 hidden_dims=(128, 128),
                 use_rnn=False,
                 rnn_hidden=32,
                 rnn_layers=1):
        super().__init__()
        self.num_moments = num_moments
        self.use_rnn = use_rnn

        if use_rnn and macro_feature_dim > 0:
            self.rnn = nn.GRU(
                input_size=macro_feature_dim,
                hidden_size=rnn_hidden,
                num_layers=rnn_layers,
                batch_first=True
            )
            macro_out_dim = rnn_hidden
        else:
            self.rnn = None
            macro_out_dim = macro_feature_dim

        in_dim = ind_feature_dim + macro_out_dim
        layers = []
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        self.mlp = nn.Sequential(*layers)
        self.last = nn.Linear(in_dim, num_moments)

    def forward(self, R, X, macro, mask):
        """
        Returns h with shape [K, T, N]
        """
        device = R.device
        T, N = R.shape

        if self.rnn is not None and macro is not None:
            macro_in = macro.unsqueeze(0)
            macro_out, _ = self.rnn(macro_in)
            macro_feat = macro_out.squeeze(0)       # [T, H]
        else:
            macro_feat = macro if macro is not None else torch.zeros(T, 0, device=device)

        if macro_feat.numel() > 0:
            macro_tiled = macro_feat.unsqueeze(1).expand(T, N, macro_feat.shape[-1])
            feat = torch.cat([X, macro_tiled], dim=2)
        else:
            feat = X

        mask_f = mask.bool()
        feat_masked = feat[mask_f]                  # [num_obs, Ftot]

        h_hidden = self.mlp(feat_masked)
        h_out = torch.tanh(self.last(h_hidden))     # [num_obs, K]

        # back to [K, T, N]
        h_full = torch.zeros((self.num_moments, T, N), device=device)
        for k in range(self.num_moments):
            tmp = torch.zeros_like(R, device=device)
            tmp[mask_f] = h_out[:, k]
            h_full[k] = tmp

        return h_full

def unconditional_loss(SDF, R, mask):
    """
    h = 1  → E[M R - 1] ≈ 0
    """
    price_err = (SDF.unsqueeze(1) * R - 1.0) * mask.float()  # [T, N]
    denom = mask.float().sum()
    mean_err = price_err.sum() / (denom + 1e-8)
    return mean_err.pow(2)


def conditional_loss(SDF, R, mask, h_full):
    """
    h_full: [K, T, N]
    L(M, ω) = E[(M R - 1) * ω]
    minimize over M, maximize over ω
    """
    price_err = (SDF.unsqueeze(1) * R - 1.0) * mask.float()  # [T, N]
    K = h_full.shape[0]

    # broadcast price_err to [K, T, N]
    pe = price_err.unsqueeze(0).expand(K, -1, -1)
    h_masked = h_full * mask.float().unsqueeze(0)

    # moment_k = average over t, i of pe * h_k
    denom = mask.float().sum()
    moments = (pe * h_masked).sum(dim=(1, 2)) / (denom + 1e-8)  # [K]
    return (moments.pow(2)).mean()


def residual_loss(w_full, R, mask):
    """
    Approximate version of their residual loss:
    how much of R is not explained by projection on w.
    """
    mask_f = mask.bool()
    w = w_full[mask_f]
    r = R[mask_f]

    # linear projection r_hat = alpha * w
    alpha = (r * w).sum() / (w.pow(2).sum() + 1e-8)
    r_hat = alpha * w

    num = (r - r_hat).pow(2).mean()
    den = (r.pow(2)).mean() + 1e-8
    return num / den


@torch.no_grad()
def eval_losses(gen, moment, R, X, macro, mask):
    gen.eval()
    moment.eval()

    SDF, w_full = gen(R, X, macro, mask)
    h = moment(R, X, macro, mask)

    loss_u = unconditional_loss(SDF, R, mask)
    loss_c = conditional_loss(SDF, R, mask, h)
    loss_r = residual_loss(w_full, R, mask)

    return loss_u.item(), loss_c.item(), loss_r.item()

@torch.no_grad()
def evaluate_pricing_errors(gen, R, X, macro, mask, device=None, name="TEST"):
    """
    R:     [T, N] numpy
    X:     [T, N, F] numpy
    macro: [T, M] numpy or None
    mask:  [T, N] numpy bool
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # to torch
    R_t = torch.tensor(R, dtype=torch.float32, device=device)
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    mask_t = torch.tensor(mask, dtype=torch.bool, device=device)
    macro_t = None
    if macro is not None:
        macro_t = torch.tensor(macro, dtype=torch.float32, device=device)

    gen.eval()
    SDF, w_full = gen(R_t, X_t, macro_t, mask_t)   # SDF: [T], w_full: [T, N]

    # pricing error: M_t R_ti - 1
    price_err = (SDF.unsqueeze(1) * R_t - 1.0) * mask_t.float()   # [T, N]

    # cross-sectional average each time
    N_t = mask_t.float().sum(dim=1)                             # [T]
    cs_mean_err = price_err.sum(dim=1) / (N_t + 1e-8)           # [T]

    mean_err = cs_mean_err.mean().item()
    std_err  = cs_mean_err.std().item()

    print(f"[{name}] mean pricing error: {mean_err:.6f}, std over time: {std_err:.6f}")
    return mean_err, std_err, cs_mean_err.cpu().numpy()

@torch.no_grad()
def evaluate_sdf_factor(gen, R, X, macro, mask, device=None, name="TEST", freq_per_year=12):
    """
    Returns time series of SDF-implied factor returns and its Sharpe.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    R_t = torch.tensor(R, dtype=torch.float32, device=device)
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    mask_t = torch.tensor(mask, dtype=torch.bool, device=device)
    macro_t = None
    if macro is not None:
        macro_t = torch.tensor(macro, dtype=torch.float32, device=device)

    gen.eval()
    SDF, w_full = gen(R_t, X_t, macro_t, mask_t)   # w_full: [T, N]

    # factor return f_t = normalized weighted return
    w_abs_sum = (w_full.abs() * mask_t.float()).sum(dim=1) + 1e-8  # [T]
    f_t = (w_full * R_t * mask_t.float()).sum(dim=1) / w_abs_sum   # [T]

    f_np = f_t.cpu().numpy()
    mean_ret = f_np.mean()
    std_ret  = f_np.std()

    if std_ret > 0:
        sharpe_ann = (mean_ret / std_ret) * np.sqrt(freq_per_year)
    else:
        sharpe_ann = 0.0

    print(f"[{name}] factor mean: {mean_ret:.6f}, std: {std_ret:.6f}, annualized Sharpe: {sharpe_ann:.3f}")
    return f_np, sharpe_ann

def train_gan_sdf(
    R_train, X_train, macro_train, mask_train,
    R_valid=None, X_valid=None, macro_valid=None, mask_valid=None,
    num_epochs_unc=50,
    num_epochs=200,
    moment_steps=10,
    lambda_res=0.5,
    lr_gen = 5e-4,
    lr_moment=1e-3,
    device="cuda" if torch.cuda.is_available() else "cpu",
    log_valid_every=10,
    return_history=True
):
    T, N, F = X_train.shape
    M = 0 if macro_train is None else macro_train.shape[-1]

    # --------- move TRAIN to torch ----------
    R_train = torch.tensor(R_train, dtype=torch.float32, device=device)
    X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
    mask_train = torch.tensor(mask_train, dtype=torch.bool, device=device)
    macro_train = None if macro_train is None else torch.tensor(macro_train, dtype=torch.float32, device=device)

    # --------- move VALID to torch (once) ----------
    if R_valid is not None:
        Rv = torch.tensor(R_valid, dtype=torch.float32, device=device)
        Xv = torch.tensor(X_valid, dtype=torch.float32, device=device)
        maskv = torch.tensor(mask_valid, dtype=torch.bool, device=device)
        macrov = None if macro_valid is None else torch.tensor(macro_valid, dtype=torch.float32, device=device)
    else:
        Rv = Xv = maskv = macrov = None

    # --------- models ----------
    gen = SDFGenerator(F, M, hidden_dims=(128, 128, 64), use_rnn=False).to(device)
    moment = MomentNet(F, M, num_moments=30, hidden_dims=(128, 128), use_rnn=False).to(device)

    opt_gen = torch.optim.Adam(gen.parameters(), lr=lr_gen)
    opt_mom = torch.optim.Adam(moment.parameters(), lr=lr_moment)

    # --------- history dict ----------
    history = {
        "unc_epoch": [],
        "unc_loss_train": [],
        "unc_loss_valid": [],
        "unc_cond_valid": [],
        "unc_res_valid": [],

        "gan_epoch": [],
        "gan_cond_train": [],
        "gan_res_train": [],
        "gan_unc_valid": [],
        "gan_cond_valid": [],
        "gan_res_valid": []
    }

    # ---------- Phase 1: unconditional training ----------
    for epoch in range(num_epochs_unc):
        gen.train()
        opt_gen.zero_grad()
        SDF, w_full = gen(R_train, X_train, macro_train, mask_train)
        loss_u = unconditional_loss(SDF, R_train, mask_train) + lambda_res * residual_loss(w_full, R_train, mask_train)
        loss_u.backward()
        opt_gen.step()

        # record train loss
        history["unc_epoch"].append(epoch + 1)
        history["unc_loss_train"].append(loss_u.item())

        # optional validation monitoring
        if Rv is not None and (epoch + 1) % log_valid_every == 0:
            val_u, val_c, val_r = eval_losses(gen, moment, Rv, Xv, macrov, maskv)
            history["unc_loss_valid"].append(val_u)
            history["unc_cond_valid"].append(val_c)
            history["unc_res_valid"].append(val_r)
            print(f"[Unc] Epoch {epoch+1}/{num_epochs_unc}, loss={loss_u.item():.4f}")
            print(f"   [Valid] u={val_u:.4f} c={val_c:.4f} r={val_r:.4f}")
        else:
            # keep alignment: fill None when not evaluated
            history["unc_loss_valid"].append(None)
            history["unc_cond_valid"].append(None)
            history["unc_res_valid"].append(None)

    # ---------- Phase 2: adversarial training ----------
    for epoch in range(num_epochs):
        # (a) update moment / discriminator: maximize conditional loss
        for _ in range(moment_steps):
            moment.train()
            gen.eval()
            opt_mom.zero_grad()
            SDF, _ = gen(R_train, X_train, macro_train, mask_train)
            h = moment(R_train, X_train, macro_train, mask_train)
            loss_m = conditional_loss(SDF, R_train, mask_train, h)
            (-loss_m).backward()     # maximize → minimize negative
            opt_mom.step()

        # (b) update generator: minimize conditional + residual
        gen.train()
        moment.eval()
        opt_gen.zero_grad()
        SDF, w_full = gen(R_train, X_train, macro_train, mask_train)
        h = moment(R_train, X_train, macro_train, mask_train)
        loss_c = conditional_loss(SDF, R_train, mask_train, h)
        loss_r = residual_loss(w_full, R_train, mask_train)
        loss_g = loss_c + lambda_res * loss_r
        loss_g.backward()
        opt_gen.step()

        # record train errors
        history["gan_epoch"].append(epoch + 1)
        history["gan_cond_train"].append(loss_c.item())
        history["gan_res_train"].append(loss_r.item())

        # validation monitoring
        if Rv is not None and (epoch + 1) % log_valid_every == 0:
            val_u, val_c, val_r = eval_losses(gen, moment, Rv, Xv, macrov, maskv)
            history["gan_unc_valid"].append(val_u)
            history["gan_cond_valid"].append(val_c)
            history["gan_res_valid"].append(val_r)

            print(f"[GAN] Epoch {epoch+1}/{num_epochs}, cond_loss={loss_c.item():.4f}, res={loss_r.item():.4f}")
            print(f"   [Valid-Unc] u={val_u:.4f} c={val_c:.4f} r={val_r:.4f}")
        else:
            history["gan_unc_valid"].append(None)
            history["gan_cond_valid"].append(None)
            history["gan_res_valid"].append(None)

    if return_history:
        return gen, moment, history, w_full
    else:
        return gen, moment, w_full


dates = date.to_numpy()  # date is your unique dates Series

train_mask_t = dates <= train_end
valid_mask_t = (dates > train_end) & (dates <= valid_end)
test_mask_t  = dates > valid_end

R_train     = R[train_mask_t]
X_train     = X[train_mask_t]
mask_train  = mask[train_mask_t]
macro_train = macro_arr[train_mask_t]

R_valid     = R[valid_mask_t]
X_valid     = X[valid_mask_t]
mask_valid  = mask[valid_mask_t]
macro_valid = macro_arr[valid_mask_t]

R_test      = R[test_mask_t]
X_test      = X[test_mask_t]
mask_test   = mask[test_mask_t]
macro_test  = macro_arr[test_mask_t]

# 1) Replace inf with nan first
X_train = X_train.copy()
macro_train = macro_train.copy()
X_valid = X_valid.copy()
macro_valid = macro_valid.copy()
X_test = X_test.copy()
macro_test = macro_test.copy()

X_train[~np.isfinite(X_train)] = np.nan
macro_train[~np.isfinite(macro_train)] = np.nan
X_valid[~np.isfinite(X_valid)] = np.nan
macro_valid[~np.isfinite(macro_valid)] = np.nan
X_test[~np.isfinite(X_test)] = np.nan
macro_test[~np.isfinite(macro_test)] = np.nan

# 2) Fill NaN/inf with 0 (or later: mean)
X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
macro_train = np.nan_to_num(macro_train, nan=0.0, posinf=0.0, neginf=0.0)
X_valid = np.nan_to_num(X_valid, nan=0.0, posinf=0.0, neginf=0.0)
macro_valid = np.nan_to_num(macro_valid, nan=0.0, posinf=0.0, neginf=0.0)
X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
macro_test = np.nan_to_num(macro_test, nan=0.0, posinf=0.0, neginf=0.0)

device = "cuda" if torch.cuda.is_available() else "cpu"

# run train
gen, moment, history, w_full  = train_gan_sdf(
    R_train, X_train, macro_train, mask_train,
    R_valid, X_valid, macro_valid, mask_valid
)

# Evaluate on TRAIN (just to see if it overfits)
evaluate_pricing_errors(gen, R_train, X_train, macro_train, mask_train, device, name="TRAIN")
evaluate_sdf_factor(gen, R_train, X_train, macro_train, mask_train, device, name="TRAIN")

# Evaluate on VALID
evaluate_pricing_errors(gen, R_valid, X_valid, macro_valid, mask_valid, device, name="VALID")
evaluate_sdf_factor(gen, R_valid, X_valid, macro_valid, mask_valid, device, name="VALID")

# Evaluate on TEST (the one we really care about)
evaluate_pricing_errors(gen, R_test, X_test, macro_test, mask_test, device, name="TEST")
factor_ret_test, sharpe_test = evaluate_sdf_factor(gen, R_test, X_test, macro_test, mask_test, device, name="TEST")

# check if the model collapse
print(w_full.mean(), w_full.std())

# UNCONDITIONAL PHASE HISTORY
history_unc = pd.DataFrame({
    "epoch": history["unc_epoch"],
    "train_unc_loss": history["unc_loss_train"],
    "valid_unc_loss": history["unc_loss_valid"],
    "valid_cond_loss": history["unc_cond_valid"],
    "valid_res_loss": history["unc_res_valid"],
})

# history_unc.to_parquet("loss_history_unc.parquet")

# GAN PHASE HISTORY
history_gan = pd.DataFrame({
    "epoch": history["gan_epoch"],
    "train_cond_loss": history["gan_cond_train"],
    "train_res_loss": history["gan_res_train"],
    "valid_unc_loss": history["gan_unc_valid"],
    "valid_cond_loss": history["gan_cond_valid"],
    "valid_res_loss": history["gan_res_valid"],
})

# history_gan.to_parquet("loss_history_gan.parquet")

