# Option Risk Manager

**Application pédagogique de valorisation d'options européennes et de gestion du risque**, développée avec Python et Streamlit dans le cadre du Master 2 IREF — Finance quantitative et actuariat.

**Version documentée : V12.8.10**

[Ouvrir l'application Streamlit](https://option-risk-manager-flzg7r8c7zohf9obxia25c.streamlit.app/)

## Objectif

Option Risk Manager permet de valoriser des positions sur options européennes avec le modèle de **Black–Scholes**, de mesurer leurs sensibilités (*Greeks*) et d'étudier la **couverture Delta-neutre** lorsque le prix du sous-jacent varie.

L'application est destinée à l'apprentissage et à l'analyse de scénarios. **Elle ne transmet aucun ordre de bourse.**

## Fonctionnalités

L'application propose quatre modes :

| Mode | Fonctionnalités principales |
| --- | --- |
| **Position individuelle** | Long Call, Short Call, Long Put, Short Put ; prix Black–Scholes, valeur de position, Delta, Gamma et autres Greeks ; couverture Delta, évolution du spot, P&L, stress tests, payoff et Risk Ladder. |
| **Portefeuille multi-options** | Agrégation des positions, de leurs valeurs et de leurs sensibilités ; comparaison des risques et couvertures à S₀ et S₁ ; P&L et stress tests du portefeuille. |
| **Monte-Carlo & couverture dynamique** | Simulation de trajectoires du sous-jacent et étude de la couverture Delta réajustée au cours du temps, avec analyse des résultats simulés. |
| **Données réelles** | Catalogue de **300 actions** cotées en **EUR ou USD**, données historiques via Yahoo Finance (`yfinance`), cours et volatilité historique annualisée, graphiques et transfert de paramètres vers les modules de simulation. |

L'interface propose également des visualisations interactives et des exports CSV sur les vues concernées.

## Comment interpréter les indications de couverture ?

**Attention : la couverture cible et l'ordre de rééquilibrage ne désignent pas la même chose.**

- **À S₀ — Couverture initiale** : nombre **total** d'actions à acheter ou vendre pour neutraliser le Delta de la position au cours initial S₀.
- **À S₁ — Couverture cible** : nombre **total** d'actions à détenir pour neutraliser le Delta après le scénario de variation du spot.
- **Rééquilibrage de S₀ vers S₁** : nombre **supplémentaire** d'actions à acheter ou vendre **uniquement si la couverture initiale à S₀ a déjà été mise en place**.

Les deux colonnes du tableau comparatif S₀/S₁ décrivent donc **deux positions cibles totales**, et **non deux ordres successifs à exécuter**. L'encadré de rééquilibrage présente seulement la différence entre ces positions.

Mathématiquement, en notant Δ₀ et Δ₁ les Delta **agrégés des options** :

```text
Couverture cible à S₀ : H₀ = −Δ₀
Couverture cible à S₁ : H₁ = −Δ₁
Rééquilibrage si H₀ est déjà détenue : H₁ − H₀
```

**Exemple :**

| Indication | Lecture |
| --- | --- |
| À S₀ : **ACHETER 98,68 actions** | Couverture totale initiale nécessaire. |
| À S₁ : **ACHETER 76,30 actions** | Nouvelle couverture totale nécessaire après le scénario. |
| Rééquilibrage : **VENDRE 22,37 actions** | Si la couverture initiale est déjà détenue, vendre environ 22,37 actions pour passer de la première position cible à la seconde. |

*Les quantités sont arrondies pour l'affichage : le calcul du rééquilibrage utilise les valeurs non arrondies.*

**Si aucune couverture n'a été mise en place à S₀**, il ne faut **pas** exécuter le rééquilibrage : il faut directement appliquer la **couverture totale cible à S₁**.

Toutes ces indications sont **théoriques et informatives** : **aucun achat ni aucune vente n'est exécuté par l'application**.

## Hypothèses et limites

- Modèle de **Black–Scholes–Merton** pour des options **européennes**, sans dividendes (`q = 0`), avec taux sans risque et volatilité supposés constants dans le cadre de la valorisation.
- Le prix calculé est un **prix théorique**, pas une cotation d'option observée en bourse.
- Les positions longues et courtes ont des sensibilités signées ; les sensibilités d'un portefeuille sont agrégées selon les quantités et les sens des positions.
- Convention de **1 action par option** dans les calculs de couverture ; ce n'est pas nécessairement le multiplicateur des contrats négociés sur les marchés.
- Les scénarios instantanés de spot et les approximations Delta/Gamma ne doivent pas être confondus avec une performance future garantie.
- Les données historiques de `yfinance` peuvent être différées ou indisponibles. La **volatilité historique** calculée à partir des cours n'est **pas** une volatilité implicite de marché.
- Les actifs sont présentés en **EUR ou USD**. **Aucune conversion de change automatique** n'est effectuée : il faut éviter de mélanger des montants dans des devises différentes.
- Le catalogue comporte 300 tickers préconfigurés ; leur disponibilité et leur devise effective restent dépendantes de Yahoo Finance.
- Les simulations Monte-Carlo dépendent de leurs hypothèses et paramètres : elles ne constituent ni une prévision certaine ni une recommandation d'investissement.

## Installation et exécution

### Prérequis

- Python et `pip`
- Les dépendances Python du dépôt (`requirements.txt`)
- Les fichiers auxiliaires de simulation Monte-Carlo présents dans le dépôt

### Depuis le dépôt GitHub

```powershell
git clone https://github.com/AnthonyIREF/Option-Risk-Manager.git
cd Option-Risk-Manager
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Vérifier que `requirements.txt` comprend notamment `streamlit`, `numpy`, `pandas`, `scipy`, `plotly` et **`yfinance`**, ainsi que les dépendances nécessaires aux modules auxiliaires.

Le programme principal importe les modules :

- `option_mc_hedging_v12.py`
- `option_mc_trace_v12_1.py`

Ils doivent être disponibles à côté de `app.py`.

### Depuis JupyterLab

Le fichier de livraison au format `.txt` peut commencer par `%%writefile app.py`, commande spécifique à une cellule Jupyter. **Cette ligne ne doit pas être copiée dans `app.py` sur GitHub.**

Dans le terminal JupyterLab :

```powershell
cd C:\Users\antho
streamlit run app.py
```

## Utilisation rapide

1. Choisir le mode **Position individuelle** ou **Portefeuille multi-options**.
2. Renseigner le spot, le strike, la maturité, le taux sans risque, la volatilité, le type d'option et la quantité.
3. Consulter le prix, les Greeks et la **couverture Delta-neutre à S₀**.
4. Modifier le spot simulé **S₁** pour observer la nouvelle exposition.
5. Lire la **couverture totale cible à S₁**, puis le **rééquilibrage** si la couverture initiale a effectivement été prise.
6. Explorer les graphiques, le P&L, les stress tests ou la simulation Monte-Carlo.

Dans **Données réelles**, sélectionner une action, charger son historique et, si souhaité, importer le spot et la volatilité historique dans les autres modes. Les autres paramètres restent à renseigner ou à vérifier manuellement.

## Technologies

Python · Streamlit · NumPy · pandas · SciPy · Plotly · yfinance

## Références

- Black, F. & Scholes, M. (1973), *The Pricing of Options and Corporate Liabilities*. https://doi.org/10.1086/260062
- Merton, R. C. (1973), *Theory of Rational Option Pricing*. https://doi.org/10.2307/3003143
- [The Options Industry Council](https://www.optionseducation.org/) — ressources pédagogiques sur les options et leurs risques.

---

**Avertissement :** projet universitaire à vocation pédagogique. Les résultats sont indicatifs, reposent sur des hypothèses simplificatrices et ne constituent pas des conseils d'investissement.
