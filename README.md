# Deep RL Exposure Control

[Français](#français) | [English](#english)

## Français

Deep RL sur séries temporelles financières. Un agent Soft Actor-Critic (stable-baselines3) ajuste l'exposition d'une stratégie de tendance sur le S&P 500 (`^GSPC`), évalué en walk-forward.

### Principe

\- Une baseline (tendance 63 jours, volatilité cible 10 %) donne une exposition `w_base`.\
\- L'agent choisit un multiplicateur `A_t dans [0, A_max]` : `w_t = clip(A_t*w_base_t, ±w_max)`.\
\- Récompense : P&L net - coûts de transaction - pénalité de risque (optionnelle).\
\- Observations : momentum 21 et 63 j, z-score des rendements, régime et ratios de volatilité, drawdown.

### Stratégies comparées

Toutes visent 10 % de volatilité avec le même plafond `w_max` ; seul le signal change.

| Stratégie | Signal |
|---|---|
| Baseline | signe du rendement sur 63 jours |
| Long only | toujours long (Zhang et al., 2020) |
| Sign(R) | signe du rendement sur 252 jours (Zhang et al., 2020) |
| MACD | moyenne de 3 signaux MACD (Zhang et al., 2020 ; Baz et al., 2015) |
| Buy & hold | une unité de l'indice, sans rebalancement |

Les ambiguïtés du papier (somme ou moyenne des MACD, définition du downside risk, demi-vie) sont tranchées d'après Lim et al. (2019) et leur code, et documentées dans `utils/`.

### Résultats

Walk-forward sur 6 ans hors échantillon (2020 à 2025) : 4 ans d'entraînement, 1 an de test, 3 seeds, 1 bp de coût par transaction.

| | Rendement/an | Volatilité | Sharpe | Max drawdown |
|---|---|---|---|---|
| SAC (3 seeds équipondérés) | 9,1 % | 11,7 % | 0,81 | -18,1 % |
| Baseline | 5,1 % | 11,2 % | 0,50 | -18,5 % |
| Long only | 8,5 % | 11,2 % | 0,79 | -14,0 % |
| Sign(R) | 6,1 % | 11,2 % | 0,59 | -15,9 % |
| MACD | 4,6 % | 7,5 % | 0,64 | -14,3 % |
| Buy & hold (hors dividendes) | 13,2 % | 20,9 % | 0,70 | -33,9 % |

\- SAC améliore nettement la baseline (Sharpe 0,81 contre 0,50) et bat la baseline sur 4 années sur 6.\
\- Face aux stratégies du papier, c'est plus serré : SAC ne bat *Long only* que 2 années sur 6.\
\- SAC trade environ 6 fois plus que la baseline. Son avantage disparaît vers 4 bp de coût aller-retour.\
\- Avec 3 seeds, l'écart entre seeds est du même ordre que l'écart avec la baseline : il en faudrait une quinzaine pour conclure.

Les 9 indicateurs du papier (Sortino, Calmar, % de jours positifs…) et le détail année par année sont dans le notebook.

### Code

```
SAC_control_project.ipynb   données, entraînement, évaluation
utils/
  market_data.py   prix (yfinance)
  features.py      observations de l'agent
  baseline.py      stratégies de référence
  environment.py   environnement Gymnasium
  scaling.py       normalisation
  metrics.py       indicateurs de performance
  runner.py        walk-forward
```

```bash
git clone https://github.com/AdemmBr/deep-rl-exposure-control.git
cd deep-rl-exposure-control
pip install -r requirements.txt
jupyter lab SAC_control_project.ipynb
```

## English

Deep RL on financial time series. A Soft Actor-Critic agent (stable-baselines3) scales the exposure of a trend strategy on the S&P 500 (`^GSPC`), evaluated walk-forward.

### How it works

\- A baseline (63-day trend, 10% volatility target) gives an exposure `w_base`.\
\- The agent picks a multiplier `A_t in [0, A_max]`: `w_t = clip(A_t*w_base_t, ±w_max)`.\
\- Reward: net P&L - transaction costs - optional risk penalty.\
\- Observations: 21 and 63-day momentum, return z-score, volatility regime and ratios, drawdown.

### Benchmarks

All target 10% volatility with the same cap `w_max`; only the signal differs.

| Strategy | Signal |
|---|---|
| Baseline | sign of the 63-day return |
| Long only | always long (Zhang et al., 2020) |
| Sign(R) | sign of the 252-day return (Zhang et al., 2020) |
| MACD | mean of 3 MACD signals (Zhang et al., 2020; Baz et al., 2015) |
| Buy & hold | one unit of the index, never rebalanced |

Ambiguities in the paper (sum vs. mean of MACD signals, downside risk definition, half-life) are settled using Lim et al. (2019) and their code, and documented in `utils/`.

### Results

Walk-forward over 6 out-of-sample years (2020 to 2025): 4 years of training, 1 year of testing, 3 seeds, 1 bp cost per trade.

| | Return/yr | Volatility | Sharpe | Max drawdown |
|---|---|---|---|---|
| SAC (3 seeds, equal-weighted) | 9.1% | 11.7% | 0.81 | -18.1% |
| Baseline | 5.1% | 11.2% | 0.50 | -18.5% |
| Long only | 8.5% | 11.2% | 0.79 | -14.0% |
| Sign(R) | 6.1% | 11.2% | 0.59 | -15.9% |
| MACD | 4.6% | 7.5% | 0.64 | -14.3% |
| Buy & hold (excl. dividends) | 13.2% | 20.9% | 0.70 | -33.9% |

\- SAC clearly improves on the baseline (Sharpe 0.81 vs 0.50) and beats it in 4 of 6 years.\
\- Against the paper's strategies it's closer: SAC beats *Long only* in only 2 of 6 years.\
\- SAC trades about 6 times more than the baseline. Its edge disappears around 4 bp round-trip cost.\
\- With 3 seeds, the spread between seeds is about as large as the gap to the baseline: around 15 would be needed to conclude.

The paper's 9 metrics (Sortino, Calmar, % positive days…) and the year-by-year breakdown are in the notebook.

### Run

```bash
git clone https://github.com/AdemmBr/deep-rl-exposure-control.git
cd deep-rl-exposure-control
pip install -r requirements.txt
jupyter lab SAC_control_project.ipynb
```

## References

\- Zhang, Z., Zohren, S., & Roberts, S. (2020). Deep Reinforcement Learning for Trading. *The Journal of Financial Data Science*, 2(2), 25–40. [arXiv:1911.10107](https://arxiv.org/abs/1911.10107)\
\- Lim, B., Zohren, S., & Roberts, S. (2019). Enhancing Time-Series Momentum Strategies Using Deep Neural Networks. *The Journal of Financial Data Science*, 1(4), 19–38. [arXiv:1904.04912](https://arxiv.org/abs/1904.04912)\
\- Baz, J., Granger, N., Harvey, C. R., Le Roux, N., & Rattray, S. (2015). Dissecting Investment Strategies in the Cross Section and Time Series. [SSRN 2695101](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2695101)\
\- Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor. *ICML*. [arXiv:1801.01290](https://arxiv.org/abs/1801.01290)\
\- Raffin, A., Hill, A., Gleave, A., Kanervisto, A., Ernestus, M., & Dormann, N. (2021). Stable-Baselines3: Reliable Reinforcement Learning Implementations. *JMLR*, 22(268), 1–8.
