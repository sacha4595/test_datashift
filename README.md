

<hr style="border-width:2px;border-color:white">
<center><h1>Rapport Test Technique Data Scientist</h1></center>
<center><h2> Mesure de la dérive </h2></center>
<hr style="border-width:2px;border-color:white">

Candidat : **Bouchez--Delotte Sacha**

# 1. Introduction

Lors de ce test technique, l'objectif est d'étudier, mesurer et analyser la dérive des données en machine learning. Pour cela nous avons à notre disposition des données sur les quatre périodes de l'année. Nous allons entraîner des modèles sur la période 0 et les tester sur les périodes 1, 2 et 3. Nous allons ensuite étudier la dérive de ces modèles sur ces périodes, ainsi que la dérive des données grâce à des métriques spécifiques.

Nous tenterons dans un dernier temps de corriger cette dérive.

# 2. Entraînement des modèles

Nous entraînons les modèles `LinearRegression` et `DecisionTreeRegressor` sur la période 0 et les testons sur les périodes 1, 2 et 3. Nous utilisons le $r^2$ score comme métrique de performance. Voici les résultats :


| Modèle | Période 1 | Période 2 | Période 3 |
|:------:|:---------:|:---------:|:---------:|
| LinearRegression | 0.745 | 0.711 | 0.563 |
| DecisionTreeRegressor | 0.790 | 0.777 | 0.692 |


Les prédictions effectuées par les modèles entraînés sur la période 0 sur les périodes suivantes témoignent bien du shift dans les données. En effet, les performances du modèle sur les périodes 1, 2 et 3 sont dégréssives.

# 3. Mesure de la dérive

## 3.1. Métriques de dérive des variables catégorielles

Pour une variable catégorielle répondant à une distribution de probabilité, on définit la divergence de Kullback-Leibler comme suit :

$$D_{KL}(P||Q) = \sum_{i=1}^{n} P(x_i) \log \frac{P(x_i)}{Q(x_i)}$$

où $P$ et $Q$ sont les distributions de probabilité de la variable catégorielle, et la divergence de Jenson-Shannon comme suit :

$$D_{JS}(P||Q) = \frac{1}{2} D_{KL}(P||M) + \frac{1}{2} D_{KL}(Q||M)$$

où $M$ est la moyenne des distributions $P$ et $Q$. Nous allons utiliser ces métriques pour mesurer la dérive des variables comme `TopCategory`. (cf les figures du notebook)

On remarque en effet une divergence croissante au fil des périodes. Cela signifie que la distribution des variables catégorielles change au fil du temps.

## 3.2. Métriques de dérive des variables quantitatives

Pour une variable quantitative, on définit la distance de Wasserstein d'ordre p comme suit :

$$W_p(P, Q) = \left( \frac{1}{n}\sum_{i=1}^n \|X_{(i)} - Y_{(i)}\|^p \right)^{1/p}$$

où $X_{(1)}, \ldots, X_{(n)}$ et $Y_{(1)}, \ldots, Y_{(n)}$ sont les statistiques d'ordre des échantillons $X$ et $Y$ respectivement. 

Nous calculons cette métrique pour les variables quantitatives normalisées selon leur moyenne et leur écart-type dans le but de les comparer entre elles. (cf les figures du notebook)

## 3.3 Mesure de la dérive des données

### 3.3.1 Matrice de corrélation

Nous avons également calculé les matrices de corrélation entre les variables sur chaque période (cf les figures du notebook). Ces différentes matrices de corrélations montrent que pour chaque période, les variables `BrowsingTime`, `Orders`, et `Age` sont fortement corrélées avec `TotalCart`, variable que l'on tente de prédire. Cela signifie que la dérive des variables `BrowsingTime`, `Orders`, et `Age` a un impact plus important sur la dérive du modèle. De plus, les coefficients de corrélations sont différents d'une période à l'autre, témoignant aussi de la dérive des données (Concept shift).

### 3.3.2 Résultats de l'analyse de la dérive des données

En comparant les distance (resp divergences) des variables catégorielles (resp quantitatives) des périodes 1, 2 et 3 par rapport à la 0, on remarque que les données s'éloignent plus ou moins de la distribution initiale. Certaines ont une dérive croissante et d'autres se rapproche à nouveau de la distribution initiale.

En particulier, les colonnes `Seniority`, `Items` ou `Orders` sont celles qui ont le plus de dérive. (Covariate shift)

La variable target `TotalCart` a une dérive croissante. (Prior Prabability shift)

De plus, la matrice de corrélation pour la période 0 montre sur sa dernière colonne (ou dernière ligne) que la variable `Seniority` est peu corrélée avec `TotalCart` que l'on tente de prédire. Cela signifie que la dérive de `Seniority` a moins d'impact sur la dérive du modèle. En revanche la dérive de `Orders` a un impact plus important sur la dérive du modèle étant donné que ces deux variables semblent corrélées.

## 3.4. Mesure de la dérive des modèles

La dérive des données a un impact sur la prédiction des modèles. Nous avons calculé les distance de Wasserstein entre les distributions des prédictions des modèles sur les périodes 1, 2 et 3 par rapport à la période 0. (cf les figures du notebook)

En effet, on remarque que cette distance s'acroît au fil des périodes. Cela signifie que les prédictions des modèles sur les périodes 1, 2 et 3 sont plus éloignées de celles sur la période 0. Cela témoigne bien de la dérive des modèles.



# 4. Correction de la dérive

En vu de cette analyse exploratoire des données, le modèle peut être mis à jour pour améliorer sa performance. Après quelques recherches, j'ai choisi deux méthodes pour améliorer le modèle :
- **Feature Removal** : Supprimer les variables qui ont le plus de dérive.
- **Kernel Mean Matching (KMM)** : Rééquilibrer les distributions des variables quantitatives.

## 4.1 Feature Removal

On peut dans un premier temps essayer de retirer les variables qui ont le plus de dérive. On va donc retirer les variables `Seniority` et `Items` car ont les a identifé comme ayant une forte dérive et une faible corrélation avec `TotalCart`. On obtient les résultats suivants :

| Modèle | Période 1 | Période 2 | Période 3 |
|:------:|:---------:|:---------:|:---------:|
| DecisionTreeRegressor avant adaptation | 0.790 | 0.777 | 0.692 |
| DecisionTreeRegressor après adaptation | 0.797 | 0.785 | 0.705 |


La suppression de ces variables a bien eu pour effet d'améliorer la performance du modèle, si on compare cette dernière figure avec celle de la section précédente (cf figure du notebook). Cependant, la performance du modèle ne s'est accru que très peu. Il est donc nécessaire d'essayer d'autres méthodes pour améliorer le modèle.

## 4.2 Importance weighting

La méthode KMM permet de rééquilibrer les distributions des variables quantitatives en attribuant des poids aux données d'entraînement. Ces poids sont calculés en fonction de la distribution des données d'entraînement et de la distribution des données de test. On peut donc utiliser cette méthode pour rééquilibrer les distributions des variables quantitatives. Le calcul des poids se ait par la minimisation du critère suivant :

$$w^* = \underset{w}{\operatorname{argmin}} \left( \frac{1}{2}w^TKw - \kappa^Tw \right)$$

avec $K_{i,j} = k(x_i,x_j)$, $x_i$, $x_j \in X_S$, $k$ la fonction noyau et 
$\displaystyle \kappa_i = \frac{n_S}{n_T} \sum_{x_j \in X_T} k(x_i,x_j)$.

On applique cette méthode pour corriger le shift de la période 2 relativement à la période 1. On obtient les résultats suivants :

| Modèle | Période 1 | Période 2 | Période 3 |
|:------:|:---------:|:---------:|:---------:|
| DecisionTreeRegressor avant adaptation | 0.790 | 0.777 | 0.692 |
| DecisionTreeRegressor après adaptation | 0.790 | 0.784 | 0.694 |

En effet, on remarque encore une légère amélioration, en particulier sur la période 2 pour laquelle on a calculé les poids optimaux.

# 5. Conclusion

Nous avons étudié la dérive des données et du modèle. Nous avons vu que la dérive des données induit une dérive du modèle qui dégrade le résultat des prédicitons au fil des périodes. Nous avons ensuite essayé d'améliorer le modèle en utilisant deux méthodes : Feature Removal et Importance weighting. Ces deux méthodes ont permis d'améliorer la performance du modèle. Cependant, la performance du modèle n'a pas été améliorée de manière significative. Il serait donc nécéssaire d'essayer d'autres approches ou d'affiner les paramètres et hyperparamèters des méthodes utilisées.


# Références

\[1\] [http://www.gatsby.ucl.ac.uk/~gretton/papers/covariateShiftChapter.pdf](Covariate Shift by Kernel Mean Matching)
