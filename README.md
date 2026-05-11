# 🎯 PromotIQ — Supervised Learning for Retail Promotion Response Classification

> End-to-end binary classification pipeline predicting customer response to a promotional campaign — comparing L1 Logistic Regression, Decision Tree, and Random Forest across annual and monthly feature sets on 6,884 clients and 323,548 monthly rows.

[![Python](https://img.shields.io/badge/Python-3.11-blue)]()
[![scikit-learn](https://img.shields.io/badge/scikit--learn-classification-orange)]()
[![Best AUC](https://img.shields.io/badge/Best%20Test%20AUC-0.764-brightgreen)]()
[![Clients](https://img.shields.io/badge/Clients-6%2C884-yellowgreen)]()
[![Course](https://img.shields.io/badge/Course-CHE1147H-red)]()

---

## 📌 Project Overview

**PromotIQ** solves a classic retail uplift problem: given a customer's transaction history, predict whether they will respond positively (label=1) or negatively (label=0) to a marketing promotion. The positive class represents only **9.4% of clients** — making this a severely imbalanced classification problem where raw accuracy is misleading and `class_weight='balanced'` is essential.

Six model configurations are evaluated across two feature families:

| Feature Set | Model | Best Params | Test AUC | Test Precision | Test Recall |
|---|---|---|---|---|---|
| **Annual** | **Logistic Regression** | C=0.0025 | **0.764** | 0.184 | 0.767 |
| Annual | Random Forest | min_samples_leaf=50 | 0.755 | 0.209 | 0.592 |
| Annual | Decision Tree | min_samples_leaf=250 | 0.750 | 0.201 | 0.685 |
| Monthly | Decision Tree | min_samples_leaf=2000 | 0.638 | 0.127 | 0.633 |
| Monthly | Random Forest | min_samples_leaf=50 | 0.647 | 0.142 | 0.497 |
| Monthly | Logistic Regression | C=0.005 | 0.626 | 0.133 | 0.532 |

**Best model: Annual Logistic Regression (C=0.0025, liblinear, balanced)** — Test AUC 0.764, highest recall (0.767), interpretable coefficients.

---

## 📊 Dataset

**Source:** Kaggle Retail Data Response + feature families from Assignment 3/4  
**Response file:** `Retail_Data_Response.csv` — 6,884 rows × 2 columns

| Dataset | Rows | Features | Class 0 | Class 1 | Imbalance |
|---|---|---|---|---|---|
| Annual (client-level) | 6,884 | 75 | 6,237 (90.6%) | 647 (9.4%) | ~10:1 |
| Monthly (client-month) | 323,548 | 29 | 293,139 (90.6%) | 30,409 (9.4%) | ~10:1 |

**Train/Test split:** 1/3 train / 2/3 test, `stratify=y`, `random_state=1147`
- Annual: 2,294 train / 4,590 test
- Monthly: 107,849 train / 215,699 test

### Feature Families Joined

| Family | Source File | Columns | Join Key |
|---|---|---|---|
| Monthly rolling | `mth_rolling_features.xlsx` | 22 (3/6/12M sum/mean/max) | CLNT_NO, ME_DT |
| Monthly day-of-week | `mth_day_counts.xlsx` | 7 (cnt_Mon–cnt_Sun) | CLNT_NO, ME_DT |
| Days since last txn | `days_since_last_txn.xlsx` | 1 | CLNT_NO, ME_DT |
| Annual aggregations | `annual features.xlsx` | 40 (8 stats × 5 years) | customer_id |
| Annual day-of-week | annual DOW file | 35 (7 days × 5 years) | customer_id |

---

## 🧠 Methodology

### Section 1.1 — Data Import & Join

**Monthly pipeline:**
1. Load `mth_rolling_features`, `mth_day_counts`, `days_since_last_txn` separately
2. Drop `Unnamed: 0` index columns; rename `CLNT_NO.1` → `CLNT_NO` in days_since
3. Inner-merge all three on `['CLNT_NO', 'ME_DT']`
4. Inner-join with response table on `CLNT_NO` (renamed from `customer_id`)
5. `fillna(0)` for rolling window NaN warm-up rows
6. Final shape: **323,548 rows × 31 columns** (including CLNT_NO, ME_DT, response)

**Annual pipeline:**
1. Load `annual features.xlsx` and annual DOW count file
2. Merge on `customer_id`; join with response
3. Drop 5 rows with NaN responses → **6,884 rows × 76 columns**

**Preprocessing:**
- `StandardScaler` fitted on training set only, applied to both splits
- Annual NaN imputed with 0 before scaling

---

### Section 1.2 — Models

#### L1 Logistic Regression (liblinear, class_weight='balanced')

**Monthly — C sweep [0.0001 → 0.005]:**

| C | Precision | Recall | Test AUC |
|---|---|---|---|
| 0.0001 | 0.131 | 0.563 | — |
| 0.001 | 0.132 | 0.540 | — |
| **0.005** | **0.133** | **0.532** | **0.626** |

Train AUC: 0.631 · Top features: `amt mean 12M` (0.437), `amt sum 12M` (0.437), `amt max 3M` (0.103)

**Annual — C sweep [0.0001 → 0.005]:**

| C | Precision | Recall | Test AUC |
|---|---|---|---|
| 0.0001 | 0.153 | 0.869 | — |
| **0.0025** | **0.184** | **0.767** | **0.764** |
| 0.005 | 0.193 | 0.734 | — |

Train AUC: 0.807 · Confusion matrix: `[[2607, 1532], [105, 346]]`  
Top features: `ann_txn_amt_sum_2014` (0.177), `ann_txn_amt_count_2014` (0.173), `ann_txn_amt_sum_2013` (0.137)

---

#### Decision Tree (class_weight='balanced', random_state=1147)

**Monthly — max_depth sweep [1 → 15]:**

| max_depth | Train AUC | Test AUC |
|---|---|---|
| 1 | 0.593 | 0.591 |
| 3 | 0.635 | 0.631 |
| **5** | **0.648** | **0.637** |
| 15 | 0.800 | 0.607 |

**Monthly — min_samples_leaf sweep [1 → 7,000]:**

| min_samples_leaf | Train AUC | Test AUC |
|---|---|---|
| 1 | 0.998 | 0.540 |
| 50 | 0.780 | 0.601 |
| 500 | 0.669 | 0.633 |
| **2,000** | **0.648** | **0.638** |
| 5,000 | 0.641 | 0.635 |

Best monthly DT: `min_samples_leaf=2000` — Train AUC: 0.648, Test AUC: 0.638, Precision: 0.127, Recall: 0.633

**Annual — max_depth sweep [1 → 15]:**

| max_depth | Train AUC | Test AUC |
|---|---|---|
| 1 | 0.699 | 0.690 |
| **3** | **0.784** | **0.742** |
| 4 | 0.818 | 0.702 |
| 15 | 0.986 | 0.571 |

Confusion matrix (depth=3): `[[2934, 1205], [154, 297]]`

**Annual — min_samples_leaf sweep [1 → 5,000]:**
Best: `min_samples_leaf=250` — Train AUC: 0.775, Test AUC: 0.750  
Precision: 0.201, Recall: 0.685 · Confusion matrix: `[[2914, 1225], [142, 309]]`  
Top features: `ann_txn_amt_sum_2014` (0.731), `ann_txn_amt_sum_2013` (0.235)

---

#### Random Forest (class_weight='balanced', random_state=1147)

**Monthly — min_samples_leaf sweep:**
Best: `min_samples_leaf=50` — Train AUC: 0.789, Test AUC: 0.647, Precision: 0.142, Recall: 0.497  
Top features: `amt sum 12M` (0.132), `amt mean 12M` (0.120), `amt max 12M` (0.090)

**Annual — min_samples_leaf sweep [1 → 200+]:**

| min_samples_leaf | Train AUC | Test AUC | Test Precision | Test Recall |
|---|---|---|---|---|
| 1 | 1.000 | 0.723 | 0.000 | 0.000 |
| 25 | 0.939 | 0.754 | 0.219 | 0.424 |
| **50** | **0.875** | **0.755** | **0.209** | **0.592** |
| 100 | 0.832 | 0.755 | 0.202 | 0.705 |

Confusion matrix (leaf=50): `[[3129, 1010], [184, 267]]`  
Top features: `ann_txn_amt_sum_2014` (0.155), `ann_txn_amt_count_2014` (0.106), `ann_txn_amt_sum_2013` (0.073)

**Annual — n_estimators sweep [5 → 500]:** 

---

### Section 1.3 — Comparison & Decision

**Full results table:**

| Feature Set | Model | Test AUC | Test Precision | Test Recall | Best Params |
|---|---|---|---|---|---|
| **Annual** | **LR (best)** | **0.764** | **0.184** | **0.767** | C=0.0025 |
| Annual | Random Forest | 0.755 | 0.209 | 0.592 | leaf=50 |
| Annual | Decision Tree | 0.750 | 0.201 | 0.685 | leaf=250 |
| Monthly | Decision Tree | 0.638 | 0.127 | 0.633 | leaf=2000 |
| Monthly | Random Forest | 0.647 | 0.142 | 0.497 | leaf=50 |
| Monthly | LR (worst) | 0.626 | 0.133 | 0.532 | C=0.005 |

**Selected model:** Annual Logistic Regression (C=0.0025)
- Highest test AUC (0.764) and highest recall (0.767) — critical for not missing positive responders
- Interpretable: feature coefficients directly readable
- No overfitting: Train AUC (0.807) / Test AUC (0.764) gap is acceptable

---

## 📁 Project Structure

```
che1147-assignment5/
├── Kush_Lad_Assignment_5_submission_.ipynb     # Original submission
├── Kush_Lad_Assignment_5_submission_.html      # HTML export
├── Kush_Lad_Assignment_5_Improved.ipynb        # ✅ Refactored (26 cells)
├── data/
│   ├── Retail_Data_Response.csv                # 6,884 client responses
│   ├── annual features.xlsx                    # 6,884 × 40 annual aggregations
│   ├── annual_transaction_counts_with_day_names_first.xlsx  # 6,889 × 35 DOW counts
│   ├── mth_rolling_features.xlsx               # 323,783 × 22 rolling windows
│   ├── mth_day_counts.xlsx                     # 323,783 × 9 monthly DOW
│   └── days_since_last_txn.xlsx                # 323,783 × 3 recency
└── README.md
```
## 📦 Dependencies

| Library | Role |
|---|---|
| `pandas` | Data loading, merging, feature table construction |
| `numpy` | Array operations, NaN handling |
| `scikit-learn` | LR, DT, RF, StandardScaler, train_test_split, metrics |
| `matplotlib` / `seaborn` | AUC plots, confusion matrices, feature importance |
| `scipy` | (available for distribution tests if needed) |
| `openpyxl` | Read `.xlsx` feature files |

---
## 💡 Key Lessons

1. **Annual beats monthly for stable promotion response** — year-level spending aggregations capture long-term customer value better than noisy 3/6/12-month windows for a once-per-campaign binary prediction task
2. **Always constrain tree depth when sweeping n_estimators** — unconstrained RF with 500 trees memorises the training set perfectly (AUC=1.0) while predicting all-majority on the test set
3. **Recall > Precision in marketing uplift** — missing a true responder (false negative) costs more than a wasted promotion (false positive); choose models with higher recall, not higher precision
4. **`class_weight='balanced'` is necessary but not sufficient** — 10:1 imbalance still keeps precision at 0.13–0.21; SMOTE or threshold calibration are the logical next steps
5. **min_samples_leaf is the most important DT/RF regulariser** — sweeping it from 1 to 7,000 shows a smooth AUC curve that plateaus, making the optimal value easy to identify
6. **Feature importance consistently points to 2014 spend** — `ann_txn_amt_sum_2014` is the top feature across DT and RF models, suggesting recent annual spending is the strongest predictor of promotion response

---

## 📚 References

- [Retail Data Response — Kaggle](https://www.kaggle.com/)
- [scikit-learn: Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [scikit-learn: Decision Tree](https://scikit-learn.org/stable/modules/tree.html)
- [scikit-learn: Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [Handling Imbalanced Classes — scikit-learn User Guide](https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html)

---

*CHE1147H — Data Mining in Engineering | University of Toronto Engineering*
