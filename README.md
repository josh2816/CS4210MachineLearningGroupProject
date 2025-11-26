# Predicting Super Bowl Winners Using Machine Learning (CS4210)

## Team: 
- **Yuan, Kate S.** — Department of Computer Science — ksyuan@cpp.edu
- **Yi, Joshua G.** — Department of Computer Science — jgyi@cpp.edu
- **Greene, Cyenadi A.** — Department of Computer Science — greene@cpp.edu
- **Velasco, Kenia A.** — Department of Computer Science — kavelasco@cpp.edu
- **Moroyoqui, Gabriela C.** — Department of Computer Science — gmoroyoqui@cpp.edu

---

## Overview
This project explores the theoretical foundations of predicting Super Bowl winners using supervised machine learning techniques. The dataset used is Kaggle’s “NFL Team Data 2003–2023,” which contains statistics for all 32 NFL teams across 21 seasons. Due to the severe class imbalance—only one team wins each year—the dataset presents an interesting challenge for machine learning models.

---
## Introduction
Predicting sports outcomes is difficult due to unpredictable factors such as player performance, coaching strategy, injuries, and random events. The Super Bowl presents an additional challenge: only one of 32 teams wins each season, creating a highly imbalanced dataset.

The aim of this project is to explore whether machine learning models can use season-level team statistics to identify patterns that predict Super Bowl winners.

---

## Dataset
**Dataset:** Kaggle — “NFL Team Data 2003–2023”  
**Rows:** 672 (32 teams × 21 seasons)  
**Attributes:** 30+ features describing team performance

### Data Characteristics
- Only ~3.1% of entries are Super Bowl winners (21 total).
- Some attributes contain missing or inconsistent values (e.g., `mov`, `ties`, `g`).
- Redundant features may be removed (e.g., wins/losses in favor of win percentage).
- Year and team name are excluded from predictive features.
- Super Bowl winners must be manually labeled as part of preprocessing.

---

## Methodology (Planned in our Paper)

### Preprocessing
- Clean missing values  
- Remove inconsistent or non-informative attributes  
- Reduce dimensionality by removing redundant features  
- Label winning teams  
- Address class imbalance via:
  - class weighting  
  - oversampling  
  - undersampling  
  - stratified cross-validation  

### Machine Learning Models
- Decision Tree (ID3 algorithm for feature importance)
- Naive Bayes
- Additional models may be added later

### Evaluation
- Stratified k-fold cross-validation by season
- Accuracy comparisons across models
- Remove inconsistent or non-informative attributes  
- Reduce dimensionality
