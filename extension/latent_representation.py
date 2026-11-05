import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils import AgentFeatures

N_DAYS = 20_000
SEGMENT = 63
LATENT = 2
EPOCHS = 40
FACTORS = ["trend", "log_vol"]


def simulate(seed: int):
    rng = np.random.default_rng(seed)
    trend = np.zeros(N_DAYS)
    log_vol = np.zeros(N_DAYS)
    for t in range(1, N_DAYS):
        trend[t] = 0.995 * trend[t - 1] + 0.20 * math.sqrt(1 - 0.995**2) * rng.normal()
        log_vol[t] = 0.98 * log_vol[t - 1] + 0.40 * math.sqrt(1 - 0.98**2) * rng.normal()
    sigma = 0.16 * np.exp(log_vol) / math.sqrt(252)
    returns = trend / 252 + sigma * rng.normal(size=N_DAYS)
    index = pd.bdate_range("1950-01-02", periods=N_DAYS)
    market = pd.DataFrame({"ret": returns, "price": np.exp(np.cumsum(returns))}, index)
    factors = pd.DataFrame({"trend": trend, "log_vol": log_vol}, index)
    features = AgentFeatures().build(market).dropna()
    factors = factors.loc[features.index]
    segments = np.arange(len(features)) // SEGMENT
    return features.to_numpy(np.float32), factors.to_numpy(), segments


def mlp(n_in, n_out, hidden=64):
    return nn.Sequential(
        nn.Linear(n_in, hidden), nn.ELU(), nn.Linear(hidden, hidden), nn.ELU(),
        nn.Linear(hidden, n_out),
    )


def log_normal(x, mu, logvar):
    return -0.5 * (math.log(2 * math.pi) + logvar + (x - mu) ** 2 / logvar.exp())


class VAE(nn.Module):
    def __init__(self, n_in, beta=6.0, n_segments=None):
        super().__init__()
        self.encoder = mlp(n_in, 2 * LATENT)
        self.decoder = mlp(LATENT, n_in)
        self.beta = beta
        self.prior = nn.Embedding(n_segments, 2 * LATENT) if n_segments else None

    def encode(self, x):
        return self.encoder(x).chunk(2, dim=1)

    def loss(self, x, u, n_data):
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        recon = 0.5 * ((self.decoder(z) - x) ** 2).sum(1)
        if self.prior is not None:
            p_mu, p_logvar = self.prior(u).chunk(2, dim=1)
            kl = 0.5 * (
                p_logvar - logvar
                + (logvar.exp() + (mu - p_mu) ** 2) / p_logvar.exp() - 1
            ).sum(1)
            return (recon + kl).mean()
        # decomposition de Chen et al., estimateur minibatch-weighted
        batch = x.shape[0]
        pairs = log_normal(z[:, None], mu[None], logvar[None])
        log_norm = math.log(batch * n_data)
        log_qz = torch.logsumexp(pairs.sum(2), 1) - log_norm
        log_qz_marginals = (torch.logsumexp(pairs, 1) - log_norm).sum(1)
        log_qz_x = log_normal(z, mu, logvar).sum(1)
        log_pz = log_normal(z, torch.zeros_like(z), torch.zeros_like(z)).sum(1)
        mutual_info = log_qz_x - log_qz
        total_corr = log_qz - log_qz_marginals
        dim_kl = log_qz_marginals - log_pz
        return (recon + mutual_info + self.beta * total_corr + dim_kl).mean()

    def latent(self, x):
        return self.encode(x)[0]


class TCL(nn.Module):
    def __init__(self, n_in, n_segments):
        super().__init__()
        self.feature = mlp(n_in, LATENT)
        self.classifier = nn.Linear(2 * LATENT, n_segments)

    def loss(self, x, u, n_data):
        h = self.feature(x)
        logits = self.classifier(torch.cat([h, h**2], 1))
        return nn.functional.cross_entropy(logits, u)

    def latent(self, x):
        return self.feature(x)


def fit(model, x, u, seed):
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x, u = torch.tensor(x), torch.tensor(u)
    for _ in range(EPOCHS):
        for batch in torch.randperm(len(x)).split(256):
            optimizer.zero_grad()
            model.loss(x[batch], u[batch], len(x)).backward()
            optimizer.step()
    return model


def pca(train, test):
    _, _, vt = np.linalg.svd(train - train.mean(0), full_matrices=False)
    return (train - train.mean(0)) @ vt[:LATENT].T, (test - train.mean(0)) @ vt[:LATENT].T


def spearman(a, b):
    rank = lambda v: pd.Series(v).rank().to_numpy()
    return abs(np.corrcoef(rank(a), rank(b))[0, 1])


def best_match(latent, factors):
    return [max(spearman(latent[:, j], factors[:, k]) for j in range(latent.shape[1]))
            for k in range(factors.shape[1])]


def linear_r2(train_x, train_y, test_x, test_y):
    add_one = lambda m: np.column_stack([m, np.ones(len(m))])
    coef, *_ = np.linalg.lstsq(add_one(train_x), train_y, rcond=None)
    residual = test_y - add_one(test_x) @ coef
    return list(1 - residual.var(0) / test_y.var(0))


def run(seed=0):
    features, factors, segments = simulate(seed)
    split = int(0.7 * len(features))
    mean, std = features[:split].mean(0), features[:split].std(0)
    x = (features - mean) / std
    x_train, x_test = x[:split], x[split:]
    f_train, f_test = factors[:split], factors[split:]
    u_train = segments[:split]
    n_seg = int(u_train.max()) + 1
    latents = {"PCA": pca(x_train, x_test)}
    torch.manual_seed(seed)
    models = {
        "beta-TCVAE": VAE(x.shape[1], beta=6.0),
        "TCL": TCL(x.shape[1], n_seg),
        "iVAE": VAE(x.shape[1], n_segments=n_seg),
    }
    for name, model in models.items():
        fit(model, x_train, u_train, seed)
        with torch.no_grad():
            latents[name] = tuple(model.latent(torch.tensor(m)).numpy() for m in (x_train, x_test))
    raw_r2 = linear_r2(x_train, f_train, x_test, f_test)
    rows = {"features brutes (7)": best_match(x_test, f_test) + raw_r2 + [np.nan] * 2}
    for name, (z_train, z_test) in latents.items():
        enriched = linear_r2(np.column_stack([x_train, z_train]), f_train,
                             np.column_stack([x_test, z_test]), f_test)
        gain = [e - r for e, r in zip(enriched, raw_r2)]
        rows[name] = best_match(z_test, f_test) + linear_r2(z_train, f_train, z_test, f_test) + gain
    columns = [f"|rho| {f}" for f in FACTORS] + [f"R2 {f}" for f in FACTORS] + [f"gain R2 brut+z {f}" for f in FACTORS]
    return pd.DataFrame.from_dict(rows, orient="index", columns=columns)


if __name__ == "__main__":
    torch.set_num_threads(1)
    results = [run(seed) for seed in range(10)]
    stacked = pd.concat(results)
    mean = stacked.groupby(level=0, sort=False).mean()
    half_width = 2.26 * stacked.groupby(level=0, sort=False).std() / np.sqrt(len(results))
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print(f"Moyenne sur {len(results)} simulations, echantillon de test (30 % final)")
    print(mean.round(3))
    print("Demi-largeur de l'IC a 95 %")
    print(half_width.round(3))
