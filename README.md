# Copperbelt GeoAI

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Folium](https://img.shields.io/badge/Folium-77B829?style=flat)

**🇫🇷** Tableau de bord **Streamlit** qui repère, avec l'algorithme **Isolation Forest**, les districts de RDC dont le profil minier sort de l'ordinaire (cuivre, or, zinc, nickel). Un premier outil de tri pour orienter l'exploration cuivre-cobalt dans le Copperbelt.

**🇬🇧** **Streamlit** dashboard that uses the **Isolation Forest** algorithm to flag DRC districts whose mining profile stands out (copper, gold, zinc, nickel). A first screening tool to help orient Cu-Co exploration in the Copperbelt.

---

## 🎯 Ce que fait l'application / What the app does

- 🇫🇷 Charge les districts de la RDC depuis le jeu de données `data/Mines_Africa_Districts_cleaned.csv`.
- 🇬🇧 Loads the DRC districts from `data/Mines_Africa_Districts_cleaned.csv`.
- 🇫🇷 Normalise les variables (`StandardScaler`) puis applique **Isolation Forest**.
- 🇬🇧 Scales the features (`StandardScaler`) then applies **Isolation Forest**.
- 🇫🇷 Affiche les districts atypiques (« Target ») sur un nuage de points cuivre / or et dans un tableau classé par nombre de mines de cuivre.
- 🇬🇧 Shows the outlying districts ("Target") on a copper / gold scatter plot and in a table ranked by copper mine count.
- 🇫🇷 Deux réglages dans la barre latérale : la **sensibilité** (5 % à 30 %) et les **métaux** analysés.
- 🇬🇧 Two sidebar settings: **sensitivity** (5% to 30%) and the **metals** analysed.

## 📊 Résultat avec les réglages par défaut / Result with default settings

Sensibilité 10 %, quatre métaux. / Sensitivity 10%, four metals.

| | |
|---|---|
| Districts dans le jeu de données / Districts in the dataset | 5 835 (47 pays / countries) |
| Districts de RDC / DRC districts | 38, dont 4 avec des mines de cuivre / 4 of them with copper mines |
| Cibles repérées / Targets flagged | 4 : Kolwezi (38 mines de cuivre), Haut-Shaba (38), Lubumbashi (3), Sankuru (3 mines d'or) |

🇫🇷 Kolwezi et Lubumbashi, deux grands pôles du Copperbelt, ressortent comme attendu : c'est une première vérification de bon sens.
🇬🇧 Kolwezi and Lubumbashi, two major Copperbelt hubs, stand out as expected: a first sanity check.

## 🗂️ Structure / Structure

```
copperbelt-geoai/
├── app/dashboard.py        # application Streamlit / Streamlit app
├── data/                   # districts miniers d'Afrique / African mining districts (CSV)
├── notebooks/              # 01_eda_mines_africa.ipynb : exploration des données / EDA
├── src/                    # chargement des données / data loading
├── requirements.txt
└── .devcontainer/          # environnement de développement / dev environment
```

## ▶️ Lancer l'application / Run the app

```bash
git clone https://github.com/harold92342/copperbelt-geoai.git
cd copperbelt-geoai
pip install -r requirements.txt
streamlit run app/dashboard.py
```

## ⚠️ Limites / Limitations

- 🇫🇷 Les données comptent des **mines par district** : ce ne sont pas des analyses géochimiques d'échantillons de sol ou de roche. Les « cibles » sont des **valeurs statistiquement atypiques**, pas des cibles d'exploration validées.
- 🇬🇧 The data counts **mines per district**: it is not geochemical assay data from soil or rock samples. The "targets" are **statistical outliers**, not validated exploration targets.
- 🇫🇷 Avec 38 districts, l'échantillon est petit, et le réglage de sensibilité (`contamination`) fixe à lui seul la part de districts signalés.
- 🇬🇧 With 38 districts the sample is small, and the sensitivity setting (`contamination`) alone sets the share of districts flagged.
- 🇫🇷 Les repères de la carte dans le notebook sont **indicatifs** (coordonnées saisies à la main), pas issus des données.
- 🇬🇧 The map markers in the notebook are **illustrative** (hand-entered coordinates), not taken from the data.
- 🇫🇷 Source du jeu de données : *à préciser*.
- 🇬🇧 Dataset source: *to be specified*.

## 🔭 Pistes d'amélioration / Next steps

- 🇫🇷 Ajouter de vraies données géochimiques ou géologiques (teneurs, failles, lithologie).
- 🇬🇧 Add real geochemical or geological data (grades, faults, lithology).
- 🇫🇷 Comparer avec d'autres méthodes (Local Outlier Factor, One-Class SVM).
- 🇬🇧 Compare with other methods (Local Outlier Factor, One-Class SVM).
- 🇫🇷 Remplacer les repères manuels par de vraies coordonnées.
- 🇬🇧 Replace hand-entered markers with real coordinates.

## 👤 Auteur / Author

**Timothée Kabwe Zumbi** — Project & Data Professional, PMP®
[LinkedIn](https://www.linkedin.com/in/timkabwe/) · [Portfolio](https://harold92342.github.io/TimPortfolio) · Licence MIT / MIT License
