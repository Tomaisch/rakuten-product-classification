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

### Phase B - Preprocessing, 23 steps
**Text (15):** merge title+description -> lowercase -> strip URLs -> strip emails ->
strip HTML tags (regex) -> strip HTML entities -> strip dimensions ("8 x 5") ->
strip bare numbers -> strip special chars (keep FR accents) -> collapse whitespace ->
tokenize (TweetTokenizer) -> **keep all languages** -> remove stopwords (FR+EN+DE) ->
min word length > 1 -> Snowball French stemming (optional for transformer).

**Vectorize / split / balance (3):** stratified train/test split with fixed seed (fit on
train only, no leakage) -> TF-IDF (1,2)-grams, 20k features, min_df=3 -> imbalance
handling (class-weight / SMOTE).

**Image (5):** load by `image_{imageid}_product_{productid}.jpg` -> autocrop white borders
-> resize 224x224 -> data augmentation (flips) -> feature extraction (EfficientNet-B0 /
frozen ResNet).

**Total: 39 steps** (16 descriptive + 23 preprocessing).

## 5. Repository structure

```
notebooks/   exploration + per-member notebooks
src/
  features/  preprocessing.py  (the shared, common preprocessing - text + image)
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
