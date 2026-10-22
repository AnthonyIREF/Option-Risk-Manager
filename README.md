# Option Risk Manager

**Application interactive de valorisation d'options européennes et d'analyse du risque financier**, développée en Python avec Streamlit dans le cadre du **Master 2 IREF — Finance quantitative et actuariat**.

**Projet réalisé par Anthony et Jason Rodriguez**  
**Version : V12.8.11**

**[Accéder directement à l'application Streamlit](https://option-risk-manager-flzg7r8c7zohf9obxia25c.streamlit.app/)** — aucune installation nécessaire pour la tester.

## Présentation du projet

Option Risk Manager est un outil pédagogique qui permet de **valoriser des options européennes**, d'étudier les principales sensibilités de leur prix aux paramètres de marché (*Greeks*) et de comprendre les mécanismes de **couverture Delta-neutre**.

L'application permet de passer d'une position individuelle à un portefeuille de plusieurs options, puis d'analyser comment sa valeur et son exposition au risque évoluent lorsque le cours du sous-jacent, la volatilité ou le temps restant jusqu'à l'échéance changent.

Elle associe ainsi **modélisation financière, visualisation interactive et analyse de scénarios**. Elle ne transmet aucun ordre de bourse.

## Fonctionnalités

| Module | Fonctionnalités principales |
| --- | --- |
| **Position individuelle** | Long Call, Short Call, Long Put et Short Put ; prix Black–Scholes, valeur de position, Delta, Gamma, Vega, Theta et Rho ; couverture Delta-neutre ; P&L, stress tests, payoff et Risk Ladder. |
| **Portefeuille multi-options** | Construction d'un portefeuille sur un même sous-jacent ; agrégation des valeurs et des Greeks ; comparaison des risques à S₀ et S₁ ; couverture cible et rééquilibrage ; graphiques et stress tests. |
| **Monte-Carlo et couverture dynamique** | Simulation de trajectoires du sous-jacent, réévaluation des options et étude d'une stratégie de couverture Delta réajustée au fil du temps. |
| **Données réelles** | Catalogue de **300 actions** en EUR ou USD ; récupération des historiques via Yahoo Finance (`yfinance`), estimation de la volatilité historique et transfert des paramètres vers les modules d'analyse. |

Selon les modules, l'interface propose des graphiques interactifs et des exports CSV.

## Démarche financière

### 1. Valoriser une position

À partir du spot, du strike, de la maturité, du taux sans risque et de la volatilité, l'application calcule un **prix théorique Black–Scholes**. Le sens de la position (long ou short) et la quantité déterminent sa valeur et ses sensibilités signées.

### 2. Mesurer les risques

Les *Greeks* permettent d'interpréter les principaux facteurs de variation de la valeur d'une option :

- **Delta** : sensibilité au cours du sous-jacent.
- **Gamma** : variation du Delta lorsque le cours du sous-jacent évolue.
- **Vega** : sensibilité à la volatilité.
- **Theta** : sensibilité au passage du temps.
- **Rho** : sensibilité au taux sans risque.

Pour un portefeuille, les sensibilités des différentes positions sont **agrégées**, en tenant compte de leur quantité et de leur sens.

### 3. Simuler et couvrir

L'utilisateur peut définir un scénario de marché à **S₁** et observer la nouvelle valeur du portefeuille, ses sensibilités et la **couverture Delta-neutre nécessaire**. Les visualisations permettent d'étudier l'évolution du risque au-delà d'un seul scénario.

## Comment lire les indications de couverture ?

**La couverture cible et le rééquilibrage sont deux informations différentes.**

- **À S₀ — Couverture initiale** : nombre **total** d'actions à acheter ou vendre pour neutraliser le Delta initial.
- **À S₁ — Couverture cible** : nouvelle position **totale** en actions nécessaire après le scénario.
- **Rééquilibrage de S₀ vers S₁** : actions **supplémentaires** à acheter ou vendre, **uniquement si la couverture initiale a déjà été mise en place**.

En notant Δ₀ et Δ₁ les Delta agrégés des positions sur options :

```text
Couverture cible à S₀ : H₀ = −Δ₀
Couverture cible à S₁ : H₁ = −Δ₁
Rééquilibrage (si H₀ est déjà détenue) : H₁ − H₀
```

**Exemple de lecture :**

| Indication affichée | Signification |
| --- | --- |
| **À S₀ : ACHETER 18,35 actions** | Position totale de couverture initiale. |
| **À S₁ : VENDRE 64,22 actions** | Nouvelle position totale cible après le scénario. |
| **Rééquilibrage : VENDRE 82,56 actions** | Ajustement nécessaire pour passer de la couverture initiale à la nouvelle position cible. |

Les quantités sont arrondies à l'affichage ; les calculs utilisent les valeurs non arrondies.

**Sans couverture initiale**, il ne faut pas effectuer l'ajustement indiqué : il faut directement constituer la **position totale cible à S₁**. Les indications sont purement théoriques : **aucun ordre n'est exécuté**.

## Exemple de scénario

Un portefeuille composé de **100 Calls longs** de strike 112 € et de **100 Puts longs** de strike 117 €, sur un sous-jacent initialement à 108 €, est étudié avec une maturité de 0,5 an, un taux sans risque de 3 % et une volatilité initiale de 20 %.

On simule ensuite un spot à **120,42 €**, une volatilité de **10 %** et **aucun temps écoulé**.

| Indicateur | Situation initiale S₀ | Scénario S₁ |
| --- | ---: | ---: |
| Valeur du portefeuille | 1 563,21 € | 1 187,11 € |
| Delta agrégé | −18,3473 | +64,2170 |
| Gamma agrégé | +5,0250 | +5,8655 |
| Couverture Delta-neutre cible | Acheter 18,35 actions | Vendre 64,22 actions |

Si la couverture initiale est détenue, le rééquilibrage consiste à **vendre 82,56 actions**. Cet exemple illustre la différence entre **position cible** et **ajustement de couverture**.

## Hypothèses et limites

- Le modèle de **Black–Scholes–Merton** est appliqué aux **options européennes**, sans dividendes (`q = 0`). Les calculs reposent notamment sur des hypothèses de taux sans risque et de volatilité constants.
- Les prix affichés sont **théoriques** et ne correspondent pas nécessairement aux cotations observées sur un marché.
- La convention retenue est **une action par option** pour les calculs de couverture ; elle ne reproduit pas nécessairement les multiplicateurs de contrats négociés en bourse.
- La **volatilité historique**, estimée à partir des cours, ne doit pas être confondue avec la **volatilité implicite** d'une option.
- Les données fournies par Yahoo Finance peuvent être différées, manquantes ou indisponibles ; la disponibilité des 300 tickers préconfigurés dépend de cette source.
- Les montants sont présentés en **EUR ou USD**, **sans conversion de change automatique**.
- Les approximations par les Greeks et les simulations Monte-Carlo dépendent des hypothèses retenues ; elles ne constituent pas des prévisions certaines.

## Utilisation de l'application

1. **Ouvrir l'application en ligne** à partir du lien en haut de cette page.
2. Choisir **Position individuelle**, **Portefeuille multi-options**, **Monte-Carlo** ou **Données réelles**.
3. Renseigner les paramètres de marché et les caractéristiques des options, ou importer un spot et une volatilité historique depuis le module de données réelles.
4. Consulter la valorisation, les Greeks et les graphiques.
5. Modifier le scénario de marché et comparer les résultats à **S₀** et **S₁**.
6. Interpréter séparément la **couverture cible** et son **rééquilibrage éventuel**.

## Exécution locale (facultative)

**L'application est accessible directement dans un navigateur, sans installation.** Pour exécuter le code source localement, avec Python installé :

```bash
git clone https://github.com/AnthonyIREF/Option-Risk-Manager.git
cd Option-Risk-Manager
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Les fichiers auxiliaires nécessaires aux simulations doivent être conservés dans le dépôt, aux côtés de `app.py`.

## Technologies utilisées

**Python · Streamlit · NumPy · pandas · SciPy · Plotly · yfinance**

## Références

- Black, F. & Scholes, M. (1973), *The Pricing of Options and Corporate Liabilities*. https://doi.org/10.1086/260062
- Merton, R. C. (1973), *Theory of Rational Option Pricing*. https://doi.org/10.2307/3003143
- [The Options Industry Council](https://www.optionseducation.org/) — ressources pédagogiques sur les options et leurs risques.

---

**Avertissement :** Les résultats sont indicatifs, reposent sur des hypothèses simplificatrices et ne constituent pas des conseils d'investissement.
