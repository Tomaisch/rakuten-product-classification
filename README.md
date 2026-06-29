# Rakuten France - Multimodal Product Classification

Classify French e-commerce product listings into **27 product-type codes** (`prdtypecode`)
from their **text** (title + description) and **product image**. DataScientest / Liora
bootcamp project, Rakuten challenge (`challengedata.ens.fr`).

---

## Workflow at a glance


```mermaid
flowchart LR
    classDef p0 fill:#cfe8ff,stroke:#1e6fd9,color:#0a2a4a;
    classDef eda fill:#d6f5d6,stroke:#2e9e2e,color:#0a3a0a;
    classDef prep fill:#ffe6cc,stroke:#e07b00,color:#5a2e00;
    classDef cpu fill:#d9f2f2,stroke:#1b8a8a,color:#053535;
    classDef gpu fill:#ece0ff,stroke:#7a3fd0,color:#2e0a5a;
    classDef valid fill:#ffd9e8,stroke:#d6336c,color:#5a0a2a;
    classDef rep fill:#fff3bf,stroke:#d9a200,color:#5a4500;
    classDef ckpt fill:#c3f0ca,stroke:#0f9d58,color:#063a1e;

    A["1 Get data<br/>download text tables + images"]:::p0
    B["2 EDA<br/>inspect files, join, profile, text lengths,<br/>missingness + MNAR test, duplicates,<br/>class imbalance 13.4:1 + chi-square,<br/>Kruskal-Wallis, languages + Wilson CIs,<br/>top tokens, encode 27 labels"]:::eda
    C["3 Clean & prepare (keep ALL rows / languages)<br/>merge title+desc, lowercase, strip URLs/emails/HTML/<br/>numbers/symbols (keep accents), tokenize,<br/>FR+EN+DE stopwords, drop 1-char, French stemming,<br/>images: crop border + resize 224"]:::prep
    D[("Clean dataset checkpoint<br/>84,916 products")]:::ckpt
    E["4 Features & split<br/>TF-IDF 1-2 grams 20k, SMOTE,<br/>dim-reduction compare (SVD/chi2/MI/NMF),<br/>t-SNE 2-D, SPLIT = train/test 80/20 (no val)"]:::cpu
    F1["5a Classical text models (CPU)<br/>Naive Bayes, LogReg, Linear SVM,<br/>Decision Tree, Random Forest, XGBoost,<br/>tuned KNN & XGBoost"]:::cpu
    F2["5b Transformer text models (GPU)<br/>CamemBERT, FlauBERT, DistilCamemBERT,<br/>XLM-RoBERTa, mBERT"]:::gpu
    F3["5c Image models (GPU)<br/>EfficientNet-B0/B3, ResNet50,<br/>ConvNeXt-tiny, ViT-small"]:::gpu
    F4["5d Multimodal fusion<br/>best text + best image (equal weight)"]:::gpu
    G[("Results checkpoint<br/>every model macro-F1")]:::ckpt
    H["6 Validate & interpret<br/>rank models, k-fold CV, per-class + confusion,<br/>SHAP, LIME, fairness by language, McNemar,<br/>ablation, SMOTE, sanity checks"]:::valid
    I["7 Report<br/>assemble figures + results"]:::rep
    J[("Shareable HTML report")]:::ckpt

    A --> B --> C --> D --> E
    E --> F1
    E --> F2
    E --> F3
    F1 --> G
    F2 --> F4
    F3 --> F4
    F4 --> G
    G --> H --> I --> J
```

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
22. Preprocessing sanity asserts (no-NaN text, all rows kept, TF-IDF width, train/test ID-overlap, zero train/test ID overlap, LabelEncoder round-trip)

### Phase C - Modeling: build / train (26 steps)
1. Feature engineering - TF-IDF (1,2)-grams, 20k, min_df=3 (fit on train only)
2. Stratified train/test split (80/20), seed 42 - no leakage (+ shape/width checks)
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


## 7. How to run (reproducible from scratch)

```bash
pip install -r requirements.txt          # core: Phases 0/A/B, classical C1-C13, Phase D (CPU is fine)
pip install -r requirements-gpu.txt      # OPTIONAL: GPU phases - transformers C14-C18, image C19-C24 (needs CUDA)
python src/data/make_dataset.py        # Phase 0: downloads CSVs (+images.zip if RAKUTEN_IMAGES_ZIP_ID set) into data/
python src/features/build_features.py  # Phase A+B: EDA figures + data/df_clean.csv
python src/models/benchmark_all_models.py  # Phase C: model zoo + dim-reduction/t-SNE figures + reports/benchmark_results.json
python src/models/validate_model.py    # Phase D: CV/SHAP/LIME/confusion/ablation figures + reports/validation_results.json
python src/make_report.py              # Phase E: one self-contained reports/rakuten_report.html (all figures + results)
```

All scripts **auto-create** their folders (`data/`, `reports/`, `reports/figures/`, `reports/logs/`, `.tmp/rakuten_cache/`). Run from the repo root. Every script is also a Jupyter file - the `#%%` markers let you open it and run **step by step**, seeing each step's text output + figure inline.

### Outputs by stage (where files land)
- **Phase 0** -> `data/X_train_update.csv`, `data/Y_train_CVw08PX.csv` (+ `data/images/` if downloaded)
- **Phase A/B** -> `reports/figures/A*.png`, `data/df_clean.csv` (the cleaned checkpoint)
- **Phase C** -> `reports/figures/C4_*.png`, `C5_*.png`, `reports/benchmark_results.json`, image caches in `.tmp/rakuten_cache/*.npy`
- **Phase D** -> `reports/figures/D*.png`, `reports/validation_results.json`
- **Phase E** -> `reports/rakuten_report.html` (figures embedded, shareable)

### Start from the uploaded checkpoint (skip Phase 0 + A + B)
This repo ships **`data/df_clean.csv.gz`** - the Phase A+B output (full 84,916 rows, already cleaned/merged/stemmed, with `text`, `text_nostem`, `y`, `prdtypecode`, `designation`, ids). A teammate can continue straight from Phase C without downloading raw data or re-running EDA/preprocessing:

```python
import pandas as pd
df = pd.read_csv('data/df_clean.csv.gz')   # pandas reads gzip directly - no manual unzip
```

Or just run the later scripts (they already load this file):
```bash
python src/models/benchmark_all_models.py   # Phase C - reads data/df_clean.csv(.gz)
python src/models/validate_model.py          # Phase D
python src/make_report.py                    # Phase E - HTML report
```
To rebuild the checkpoint from scratch instead, delete it and run Phase 0 + A + B.

### Resume from checkpoints (skip expensive stages)
The pipeline is checkpointed so you can start mid-way without redoing finished work:
- If `data/df_clean.csv` exists, skip Phase 0+A+B and start at Phase C.
- `benchmark_all_models.py` **resumes**: it loads `reports/benchmark_results.json` and skips any model already computed (re-running continues the slow BayesSearch / image / transformer models where it left off). Delete the JSON to force a full rebuild.
- Image features cache to `.tmp/rakuten_cache/*.npy` and are reused if present.
- GPU phases (transformers C14-C18, image models C19-C23, image extraction C24) **skip gracefully** if `torch`/GPU is absent, so the text pipeline runs on CPU alone.


## 6. Results (2-way split: train 67,932 / test 16,984, stratified seed 42, macro-F1 on test)

*Run in progress — KNN/XGB-BayesSearch still finishing; FlauBERT (collapsed under fp16) being re-run in fp32; DistilCamemBERT to be re-run. Updated live.*

**Classical text (TF-IDF):** LinearSVC **0.791** | XGBoost 0.758 | LogReg 0.783 | RandomForest 0.740 | NaiveBayes 0.682 | DecisionTree 0.583 | KNN/XGB-Bayes (running)

**Transformer text:** CamemBERT **0.831** | mBERT 0.810 | XLM-RoBERTa 0.807 | FlauBERT (re-running) | DistilCamemBERT (re-running)

**Image (fine-tuned):** ConvNeXt-tiny **0.702** | ViT-small 0.677 | EfficientNet-B3 0.624 | EfficientNet-B0 0.602 | ResNet50 0.563

**Multimodal:** CamemBERT + ConvNeXt-tiny, equal-weight late fusion = **0.855**

| Family | Best model | macro-F1 |
|---|---|---|
| Classical text | LinearSVC | 0.791 |
| Transformer text | CamemBERT | 0.831 |
| Image | ConvNeXt-tiny | 0.702 |
| **Multimodal fusion (best-of-all)** | **CamemBERT + ConvNeXt-tiny** | **0.855** |

Figures in `reports/figures/`: EDA (A7-A16), C4 dim-reduction, C5 t-SNE, C24 image models, D3-D13 (validation / SHAP / confusion / ablation).
