# Rakuten France - Multimodal Product Classification

Classify French e-commerce product listings into **27 product-type codes** (`prdtypecode`)
from their **text** (title + description) and **product image**. DataScientest / Liora
bootcamp project, Rakuten challenge (`challengedata.ens.fr`).

---

## 1. Goal

Given a product's `designation` (title), `description`, and `image`, predict its
`prdtypecode` (one of 27 commercial categories). This is a **multimodal, multi-class,
heavily imbalanced** text+image classification problem. Performance is judged on
**macro-F1** (every class weighted equally), not accuracy, because the classes are
very unevenly sized.

## 2. Research questions

Open questions the benchmark answers empirically - no answer is assumed in advance.

| # | Question | Answer (from results) |
|---|---|---|
| Q1 | Is **text or image** the stronger signal, and does **text+image fusion** beat the best single modality? |  |
| Q2 | Among **classical text models** (NB, LogReg, LinearSVC, trees, XGBoost, KNN), which fits sparse TF-IDF best? |  |
| Q3 | Among **transformers**, does a **French specialist (CamemBERT/FlauBERT)** beat a **multilingual model (XLM-RoBERTa/mBERT)** given the ~20% non-French tail? |  |
| Q4 | Among **image backbones** (EfficientNet-B0/B3, ResNet50, ConvNeXt, ViT), which inductive bias fits product photos best? |  |
| Q5 | Does **keeping all languages** beat dropping non-FR/EN rows, and does **SMOTE** beat class-weighting for the 13.4:1 imbalance (ranked by **macro-F1**)? |  |

## 3. Dataset

| Property | Value |
|---|---|
| Training products | **84,916** |
| Test products (challenge) | **13,812** |
| Target classes (`prdtypecode`) | **27** |
| Class imbalance | **13.4 : 1** (largest `2583 Pool accessories` n=10,209 / smallest `1180 Trading figures` n=764) |
| `description` missing | **35.1%** (`designation` never missing) |
| Median combined text length | **314 characters** |
| Dominant language | French (~80%), tail: EN, DE, ES, NL, IT |
| Files | `X_train_update.csv`, `Y_train_CVw08PX.csv`, `X_test_update.csv`, `images.zip` (~2.5 GB) |
| Image naming | `image_{imageid}_product_{productid}.jpg` |

The only usable numeric fields are IDs - **all the predictive signal is in the text and
the image**.

## 4. Pipeline steps

Four phases, each mapped to one script:

| Phase | What it does | Script |
|---|---|---|
| **Phase 0** | Download data (CSVs + image archive) into `data/` | `src/data/make_dataset.py` |
| **Phase A** | Descriptive analysis / EDA (18 steps) + data checks + statistical tests | `src/features/build_features.py` |
| **Phase B** | Preprocessing - text + image (22 steps) + step-demo + sanity asserts | `src/features/build_features.py` |
| **Phase C** | Modeling - build/train all candidate models (26 steps) | `src/models/benchmark_all_models.py` |
| **Phase D** | Model validation - CV, metrics, interpretability, significance, ablation (16 steps) | `src/models/validate_model.py` |

### Phase 0 - Download
1. Download `X_train_update.csv`, `Y_train_CVw08PX.csv`, `images.zip` (~2.5 GB) into `data/`; unzip images.

### Phase A - Descriptive analysis / EDA (18 steps)
1. Inspect X and Y SEPARATELY before merge (shape/head/dtypes; `len(X)==len(Y)`; row-key identity assert)
2. Merge X+Y, fillna, build combined text
3. View the data (first records)
4. Structure - `info()`, dtypes, cardinality (`nunique`)
5. Descriptive stats - char-length, word counts (`describe`, quartiles)
6. Missing values per column
7. Missing-description rate by category - MNAR (chi-square contingency + Cramer's V)
8. Duplicate analysis (exact + text-subset)
9. Class distribution + imbalance 13.4:1 [figure]
10. Chi-square goodness-of-fit on class balance
11. Cumulative imbalance curve [figure]
12. Text-length distribution [figure]
13. Kruskal-Wallis - text length across classes
14. Language detection + distribution + Wilson CIs [figure]
15. Top tokens / vocabulary size [figure]
16. Bivariate - median word count per class [figure]
17. Target encoding (27 codes -> 0..26)
18. Decisions taken from the EDA

### Phase B - Preprocessing (22 steps; KEEPS ALL ROWS / LANGUAGES)
Text:
1. Merge `designation` + `description` (fillna `''`)
2. Lowercase
3. Strip URLs
4. Strip emails
5. Strip HTML tags (BeautifulSoup)
6. Strip HTML entities (`&eacute;`)
7. Strip dimensions ("8 x 5")
8. Strip bare numbers
9. Strip special chars (keep French accents)
10. Collapse whitespace
11. Tokenize (TweetTokenizer)
12. Keep all languages (no row dropping)
13. Remove stopwords (FR + EN + DE)
14. Min word length > 1
15. Snowball French stemming (classical branch; off for transformers)
16. Step-by-step demo on one real product (raw -> final)
17. Vocabulary effect (avg tokens before/after cleaning)
Image:
18. Load image by `image_{imageid}_product_{productid}.jpg`
19. Corner-threshold crop of the white border
20. Resize 224x224
21. Data augmentation - flips (train only)
Sanity:
22. Preprocessing sanity asserts (no-NaN text, all rows kept, TF-IDF width, KL-divergence train||val, zero train/test ID overlap, LabelEncoder round-trip)

### Phase C - Modeling: build / train (26 steps)
1. Feature engineering - TF-IDF (1,2)-grams, 20k, min_df=3 (fit on train only)
2. Stratified train/val/test split, seed 42 - no leakage (+ shape/width checks)
3. Imbalance handling - class-weight or SMOTE; rank by macro-F1
4. Dimension-reduction comparison - full TF-IDF vs TruncatedSVD-300/1000 vs chi2 SelectKBest-2000/5000 vs mutual-info-2000 vs PCA vs NMF
5. t-SNE 2-D visualization
6. Naive Bayes
7. Logistic Regression
8. LinearSVC
9. Decision Tree
10. Random Forest
11. XGBoost
12. KNN - BayesSearchCV + SMOTE (MCC)
13. XGBoost - BayesSearchCV + SMOTE (MCC)
14. CamemBERT
15. FlauBERT
16. XLM-RoBERTa
17. mBERT
18. DistilCamemBERT
19. EfficientNet-B0
20. ResNet50
21. EfficientNet-B3
22. ConvNeXt-tiny
23. ViT-small
24. Image feature extraction (frozen ResNet18, 512-d, L2-normalized)
25. Multimodal fusion - concat text + image (resize vs autocrop)
26. Weighted late fusion (weight tuned on validation)

### Phase D - Model validation (16 steps)
1. Model-comparison ranking [figure]
2. K-fold cross-validation (StratifiedKFold 3) - mean +/- std
3. Per-class metrics (classification_report)
4. Confusion matrix [figure] + top confusions
5. Spearman - class support vs F1
6. Coefficients per class (inherent interpretability)
7. SHAP global importance [figure]
8. SHAP local waterfalls - 3 correct + 1 wrong [figures]
9. SHAP beeswarm
10. LIME local check (independent)
11. Fairness - accuracy per detected language
12. McNemar significance test (multimodal gain)
13. Ablation study - remove each step, measure delta macro-F1 [figure]
14. Imbalance experiments - SMOTE, SMOTE-Tomek, per-class threshold tuning
15. Debugging sanity checks - majority baseline, shuffle-labels, memorize-tiny, inspect predictions
16. Literature / SOTA + recommended methods (implemented)

**Phases at a glance:** 0 = download - A = describe (18) - B = clean (22) - C = build models (26) - D = validate models (16).

## 5. Repository structure

```
notebooks/   exploration + per-member notebooks
src/
  data/         make_dataset.py        (Phase 0 - download)
  features/     build_features.py      (Phase A EDA + Phase B preprocessing)
  models/       benchmark_all_models.py (Phase C - build/train)
                validate_model.py       (Phase D - validation/interpretability)
  streamlit/    app.py                  (defense demo)
reports/        figures/ + result JSONs + Data-Audit table
models/         trained model artifacts
references/     bibliography
```

## 6. Models benchmarked (full 84,916-corpus, common RS=42 split, macro-F1)

Every model is justified in `MODEL_SELECTION_JUSTIFICATION.md` (each tests a specific hypothesis
about the data). All trained from scratch, no cache. Results filled in after the benchmark run.

**Text - classical (TF-IDF):** Naive Bayes · Logistic Regression · LinearSVC · Decision Tree ·
Random Forest · XGBoost · KNN (BayesSearchCV + SMOTE) · XGBoost (BayesSearchCV + SMOTE)

**Text - transformers:** CamemBERT · FlauBERT · XLM-RoBERTa · mBERT · DistilCamemBERT

**Image:** EfficientNet-B0 · ResNet50 · EfficientNet-B3 · ConvNeXt-tiny · ViT-small

**Fusion:** best text + best image, late fusion, weight tuned on validation.

| Family | Best model | macro-F1 | Accuracy |
|---|---|---|---|
| Classical text |  |  |  |
| Transformer text |  |  |  |
| Image |  |  |  |
| **Multimodal fusion (best-of-all)** |  |  |  |
