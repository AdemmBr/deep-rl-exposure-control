# Deep RL Exposure Control

[Français](#français) | [English](#english)

## Français

Deep RL pour les séries temporelles, appliqué aux séries temporelles financières. Un agent de Deep Reinforcement Learning (SAC) ajuste l'exposition d'une stratégie de tendance sur le S&P 500.

Une baseline de tendance (signe du rendement sur 63 jours, mis à l'échelle pour viser 10 % de volatilité) donne une exposition de référence `w_base`. Chaque jour, l'agent choisit un multiplicateur `A_t` entre 0 et `A_max`, et l'exposition finale est `w_t = clip(A_t * w_base_t, -w_max, w_max)` : on multiplie la baseline puis on borne le résultat à `w_max` en valeur absolue. Avec `A_t = 1`, on retrouve la baseline.

### Stratégies comparées

Toutes visent 10 % de volatilité avec le même plafond `w_max` ; seul le signal change.

| Stratégie | Signal |
|---|---|
| Baseline | signe du rendement sur 63 jours, avec une volatilité cible de 10 % |
| Long only | toujours long (Zhang et al., 2020) |
| Sign(R) | signe du rendement sur 252 jours (Zhang et al., 2020) |
| MACD | moyenne de 3 signaux MACD (Zhang et al., 2020 ; Baz et al., 2015) |
| Buy & hold | une unité de l'indice, sans rebalancement |

Les ambiguïtés du papier (somme ou moyenne des MACD, définition du downside risk, demi-vie) sont tranchées d'après Lim et al. (2019) et leur code, et documentées dans `utils/`.

### Résultats

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

### Extension : représentation latente de l'état

Idéalement, deux états proches devraient correspondre à des situations de marché proches, donc à des récompenses et des valeurs Q proches. Ce n'est pas le cas ici : les features mélangent les facteurs sous-jacents (tendance, volatilité), et la distance entre deux états ne reflète pas la proximité des régimes de marché. C'est au critic et à la politique de démêler eux-mêmes cette structure.

Pour voir si une représentation apprise pouvait corriger ce défaut, on a simulé des rendements à partir de facteurs connus, calculé les mêmes features que dans le projet, puis cherché à retrouver ces facteurs avec deux familles de méthodes : le démêlage non supervisé (β-TCVAE, Chen et al., 2018) et les méthodes rendues identifiables par la non-stationnarité (TCL, Hyvärinen et Morioka, 2016 ; iVAE, Khemakhem et al., 2020).

Le résultat est négatif. Aucune méthode ne retrouve les facteurs mieux que les features brutes, et enrichir l'état n'aide que pour la volatilité, la tendance restant trop bruitée. SAC garde donc les features d'origine.

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

Deep RL for time series, applied to financial time series. A Deep Reinforcement Learning agent (SAC) scales the exposure of a trend strategy on the S&P 500.

A trend baseline (sign of the 63-day return, scaled to target 10% volatility) gives a reference exposure `w_base`. Each day the agent picks a multiplier `A_t` between 0 and `A_max`, and the final exposure is `w_t = clip(A_t * w_base_t, -w_max, w_max)`: the baseline is scaled, then capped at `w_max` in absolute value. With `A_t = 1`, you get the baseline back.

### Benchmarks

All target 10% volatility with the same cap `w_max`; only the signal differs.

| Strategy | Signal |
|---|---|
| Baseline | sign of the 63-day return, with a 10% volatility target |
| Long only | always long (Zhang et al., 2020) |
| Sign(R) | sign of the 252-day return (Zhang et al., 2020) |
| MACD | mean of 3 MACD signals (Zhang et al., 2020; Baz et al., 2015) |
| Buy & hold | one unit of the index, never rebalanced |

Ambiguities in the paper (sum vs. mean of MACD signals, downside risk definition, half-life) are settled using Lim et al. (2019) and their code, and documented in `utils/`.

### Results

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

### Extension: latent state representation

Ideally, two nearby states should correspond to similar market situations, and therefore to similar rewards and Q-values. That isn't the case here: the features mix the underlying factors (trend, volatility), so the distance between two states doesn't reflect how close the market regimes are. The critic and the policy have to untangle that structure on their own.

To see whether a learned representation could fix this, we simulated returns from known factors, computed the same features as in the project, then tried to recover those factors with two families of methods: unsupervised disentanglement (β-TCVAE, Chen et al., 2018) and methods made identifiable by non-stationarity (TCL, Hyvärinen and Morioka, 2016; iVAE, Khemakhem et al., 2020).

The result is negative. No method recovers the factors better than the raw features, and enriching the state only helps for volatility, as the trend stays too noisy. SAC therefore keeps the original features.

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
\- Raffin, A., Hill, A., Gleave, A., Kanervisto, A., Ernestus, M., & Dormann, N. (2021). Stable-Baselines3: Reliable Reinforcement Learning Implementations. *JMLR*, 22(268), 1–8.\
\- Chen, R. T. Q., Li, X., Grosse, R., & Duvenaud, D. (2018). Isolating Sources of Disentanglement in Variational Autoencoders. *NeurIPS*. [arXiv:1802.04942](https://arxiv.org/abs/1802.04942)\
\- Hyvärinen, A., & Morioka, H. (2016). Unsupervised Feature Extraction by Time-Contrastive Learning and Nonlinear ICA. *NeurIPS*. [arXiv:1605.06336](https://arxiv.org/abs/1605.06336)\
\- Khemakhem, I., Kingma, D., Monti, R., & Hyvärinen, A. (2020). Variational Autoencoders and Nonlinear ICA: A Unifying Framework. *AISTATS*. [arXiv:1907.04809](https://arxiv.org/abs/1907.04809)
