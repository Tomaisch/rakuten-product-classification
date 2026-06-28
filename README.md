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

Each phase maps to one script:

| Phase | What it does | Script |
|---|---|---|
| **Phase 0** | Download the data (text CSVs + image archive) into `data/` | `src/data/make_dataset.py` |
| **Phase A** | Descriptive analysis / EDA (16 steps, A1-A16) | `src/features/build_features.py` |
| **Phase B** | Preprocessing - text B1-B15 in `build_features.py`; image B16-B17 in the benchmark dataloader | `src/features/build_features.py` (+ `benchmark_all_models.py` for image) |
| **Phase C** | Modeling - 22 steps (18 models + setup + fusion) | `benchmark_all_models.py` |

### Phase 0 - Download
1. Download `X_train_update.csv`, `Y_train_CVw08PX.csv`, and `images.zip` (~2.5 GB) into `data/`, unzip images. Script: `make_dataset.py`.

### Phase A - Descriptive analysis (EDA), 16 steps
1. View the data (first records)
2. Structure - shape, dtypes, cardinality
3. Descriptive stats - title/description char-length, word counts
4. Missing values per column
5. Missing-description rate by category (MNAR check)
6. Class distribution + imbalance (13.4:1)
7. Chi-square goodness-of-fit on class balance
8. Cumulative imbalance curve
9. Text-length distribution
10. Kruskal-Wallis - text length across classes
11. Language detection + distribution (+ Wilson CIs)
12. Top tokens / words-by-class
13. Duplicate analysis
14. Image inspection - dimensions, white-border / blank-area detection
15. Target encoding (27 codes -> 0..26)
16. Decisions taken from the EDA

### Phase B - Preprocessing (data cleaning only - NO models here), 17 steps

Text:
1. Merge `designation` + `description` (fillna `''`)
2. Lowercase
3. Strip URLs
4. Strip emails
5. Strip HTML tags (regex)
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

Image:
16. Corner-threshold crop of the white border
17. Resize 224x224

Scripts: `build_features.py` (Phase A descriptive A1-A16 + Phase B text B1-B15). Phase B image steps B16-B17 (corner-crop, resize 224) are applied in `benchmark_all_models.py`'s image dataloader.

### Phase C - Modeling (the models live here - NOT preprocessing), 22 steps
1. Stratified train/val/test split, seed 42 (fit on train only - no leakage)
2. Feature engineering - TF-IDF (1,2)-grams, 20k, min_df=3 (classical); raw merged text (transformers); image tensors (image models)
3. Imbalance handling - class-weight or SMOTE; rank by macro-F1
4. Naive Bayes (classical text)
5. Logistic Regression (classical text)
6. LinearSVC (classical text)
7. Decision Tree (classical text)
8. Random Forest (classical text)
9. XGBoost (classical text)
10. KNN - BayesSearchCV + SMOTE (classical text)
11. XGBoost - BayesSearchCV + SMOTE (classical text)
12. CamemBERT (transformer text)
13. FlauBERT (transformer text)
14. XLM-RoBERTa (transformer text)
15. mBERT (transformer text)
16. DistilCamemBERT (transformer text)
17. EfficientNet-B0 (image)
18. ResNet50 (image)
19. EfficientNet-B3 (image)
20. ConvNeXt-tiny (image)
21. ViT-small (image)
22. Fusion - best validation text model + best validation image model, late fusion weight tuned on validation, reported on test

Script: `benchmark_all_models.py`. Per-model justification in `MODEL_SELECTION_JUSTIFICATION.md`.

**Phases at a glance:** A = describe the data (16 steps) · B = clean the data (17 steps) · C = model the data (22 steps).

## 5. Repository structure

```
notebooks/   exploration + per-member notebooks
src/
  models/    text_model.py, image_model.py, fusion_model.py
  streamlit/ app.py  (defense demo)
reports/     EDA report + figures + Data-Audit table
models/       trained model artifacts
references/   bibliography
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
