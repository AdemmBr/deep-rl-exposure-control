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
\- Ces chiffres sont sur 3 seeds, le modèle a été validé sur 10 (voir plus bas).

### Choix et interprétation

Les hyperparamètres ont été testés sur GPU, puis on n'a retenu que les plus significatifs pour refaire une validation croisée dessus. On s'est volontairement limités. Avec trop d'hyperparamètres, on finit par chercher de l'interprétation partout, et sur six folds il y a toujours une combinaison qui marche par hasard.

La validation croisée a tourné sur 10 seeds et l'entraînement était stable sur chaque fold. La moyenne bougeait peu par rapport à 3 seeds, donc le notebook n'en garde que 3, ce qui fait une demi-heure de calcul sur CPU au lieu de deux heures. Un seed pris seul reste bruité (en 2025, le Sharpe va de 1,42 à 2,37 selon le seed), donc on regarde toujours la moyenne.

Pour le réseau, le meilleur résultat était un 64x64 pour l'acteur et le critic. Petit défaut du projet, on n'a testé que des MLP et pas d'architectures vraiment différentes comme un LSTM ou un transformer.

Le paramètre qui a le plus joué, c'est la fenêtre de momentum de la baseline, et c'est assez logique. Avec une fenêtre plus courte le signal réagit plus vite, mais il réagit aussi au bruit. Il se retourne plus souvent, la stratégie paie plus de coûts et se fait prendre à contre-pied, donc la performance se dégrade. 63 jours est un bon compromis.

Gamma à 0,96 correspond à un horizon d'environ 25 jours, à peu près un mois de bourse. Ça laisse à l'agent le temps de voir l'effet d'une tendance sans aller chercher trop loin. `lambda_risk = 2`, en revanche, a été fixé à partir d'un rendement moyen estimé ex post, et ce n'est pas très propre.

Quand on regarde ce que fait l'agent, le multiplicateur moyen descend sous 1 les années de crise (0,73 en 2020, 0,55 en 2022) et passe à 1,36 en 2021, année calme et haussière. En gros, SAC apprend surtout à couper l'exposition quand le marché s'agite, plus qu'à retoucher la baseline au jour le jour.

On a aussi comparé avec PPO, qui n'est pas dans ce repo. Sa performance se dégradait sur les derniers folds, et on a un peu de mal à expliquer cette différence avec SAC. Notre meilleure hypothèse, c'est l'entropie. SAC optimise la récompense plus un bonus d'entropie réglé automatiquement, donc sa politique reste aléatoire pendant l'entraînement et continue d'explorer tout l'intervalle [0, A_max]. PPO n'a pas de bonus d'entropie par défaut dans Stable-Baselines3, et comme il est on-policy il jette ses données après chaque mise à jour. Sur un millier de jours de train, sa politique finit par coller à ce qui marchait dans la fenêtre d'entraînement, et ça tient moins bien quand le régime change.

### Extension : représentation latente de l'état

Idéalement, deux états proches devraient correspondre à des situations de marché proches, donc à des récompenses et des valeurs Q proches. Ce n'est pas le cas ici : les features mélangent les facteurs sous-jacents (tendance, volatilité), et la distance entre deux états ne reflète pas la proximité des régimes de marché. C'est au critic et à la politique de démêler eux-mêmes cette structure.

Pour voir si une représentation apprise pouvait corriger ce défaut, on a simulé des rendements à partir de facteurs connus, calculé les mêmes features que dans le projet, puis cherché à retrouver ces facteurs avec deux familles de méthodes : le démêlage non supervisé (β-TCVAE, Chen et al., 2018) et les méthodes rendues identifiables par la non-stationnarité (TCL, Hyvärinen et Morioka, 2016 ; iVAE, Khemakhem et al., 2020).

Le résultat est négatif. Aucune méthode ne retrouve les facteurs mieux que les features brutes, et enrichir l'état n'aide que pour la volatilité, la tendance restant trop bruitée. SAC garde donc les features d'origine.

Sur 10 simulations, le gain de R² sur la volatilité quand on ajoute les latents aux features reste sous 0,01 (0,007 ± 0,007 pour TCL, 0,004 ± 0,004 pour iVAE, IC à 95 %), donc à la limite du significatif, et il est nul pour la tendance. Le code est dans `extension/latent_representation.py` et tourne en quatre minutes environ sur CPU. Par contre, il ne mesure que la récupération des facteurs. Pour des soucis de calcul, on n'a pas réentraîné SAC avec un état enrichi par ces latents. Vu ce que donnent les données synthétiques on ne s'attend pas à grand-chose, mais ça reste à tester.

### Code

```
SAC_control_project.ipynb   données, entraînement, évaluation
extension/
  latent_representation.py   facteurs synthétiques, β-TCVAE, TCL, iVAE
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
python extension/latent_representation.py
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
\- These figures are on 3 seeds, the model was validated on 10 (see below).

### Choices and interpretation

Hyperparameters were tested on GPU, and we only kept the most significant ones to run a cross-validation on. We deliberately limited ourselves. With too many hyperparameters you end up looking for an interpretation everywhere, and over six folds there's always some combination that works by chance.

The cross-validation ran on 10 seeds and training was stable on every fold. The average barely moved compared with 3 seeds, so the notebook only keeps 3, which means half an hour of compute on CPU instead of two hours. A single seed is still noisy (in 2025 the Sharpe goes from 1.42 to 2.37 depending on the seed), so we always look at the average.

For the network, the best result came from a 64x64 actor and critic. A small flaw of the project is that we only tried MLPs, not genuinely different architectures like an LSTM or a transformer.

The parameter that mattered most is the baseline's momentum window, which makes sense. With a shorter window the signal reacts faster, but it also reacts to noise. It flips more often, the strategy pays more costs and gets whipsawed, so performance degrades. 63 days is a good compromise.

A gamma of 0.96 means a horizon of about 25 days, roughly one trading month. That gives the agent time to see a trend play out without looking too far ahead. `lambda_risk = 2`, on the other hand, was set from a mean return estimated ex post, which isn't very clean.

Looking at what the agent actually does, the average multiplier drops below 1 in crisis years (0.73 in 2020, 0.55 in 2022) and goes up to 1.36 in 2021, a calm bull year. Basically, SAC mostly learns to cut exposure when the market gets rough, rather than tweaking the baseline day to day.

We also compared against PPO, which isn't in this repo. Its performance degraded on the last folds, and we struggle a bit to explain the difference with SAC. Our best guess is entropy. SAC optimises reward plus an automatically tuned entropy bonus, so its policy stays stochastic during training and keeps exploring the whole [0, A_max] range. PPO has no entropy bonus by default in Stable-Baselines3, and being on-policy it throws its data away after each update. Over a thousand or so training days, its policy ends up sticking to what worked in the training window, and that holds up less well when the regime changes.

### Extension: latent state representation

Ideally, two nearby states should correspond to similar market situations, and therefore to similar rewards and Q-values. That isn't the case here: the features mix the underlying factors (trend, volatility), so the distance between two states doesn't reflect how close the market regimes are. The critic and the policy have to untangle that structure on their own.

To see whether a learned representation could fix this, we simulated returns from known factors, computed the same features as in the project, then tried to recover those factors with two families of methods: unsupervised disentanglement (β-TCVAE, Chen et al., 2018) and methods made identifiable by non-stationarity (TCL, Hyvärinen and Morioka, 2016; iVAE, Khemakhem et al., 2020).

The result is negative. No method recovers the factors better than the raw features, and enriching the state only helps for volatility, as the trend stays too noisy. SAC therefore keeps the original features.

Over 10 simulations, the R² gain on volatility from adding the latents to the features stays below 0.01 (0.007 ± 0.007 for TCL, 0.004 ± 0.004 for iVAE, 95% CI), so it's borderline significant, and there's no gain at all on the trend. The code is in `extension/latent_representation.py` and runs in about four minutes on CPU. It only measures factor recovery, though. Because of compute constraints, we didn't retrain SAC with a state enriched by these latents. Given the synthetic results we don't expect much, but it's still worth testing.

### Run

```bash
git clone https://github.com/AdemmBr/deep-rl-exposure-control.git
cd deep-rl-exposure-control
pip install -r requirements.txt
jupyter lab SAC_control_project.ipynb
python extension/latent_representation.py
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
