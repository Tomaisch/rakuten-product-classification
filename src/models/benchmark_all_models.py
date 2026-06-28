# =============================================================================
# PHASE C - MODELING (22 steps). Each step/model is its own independent #%% cell.
# Steps follow PREPROCESSING_STEPS.md. Reproducible: full corpus, ONE seed=42 split.
# Input: data/df_clean.csv (produced by build_features.py).
# =============================================================================

#%% [imports + load clean data]
import numpy as np, pandas as pd
RS = 42; DATA = 'data'
df = pd.read_csv(f'{DATA}/df_clean.csv')
df['text'] = df['text'].fillna(''); df['text_nostem'] = df['text_nostem'].fillna('')

# PHASE C - MODELING (22 steps)
# =============================================================================
#%% [C1] stratified train/val/test split, seed 42 (fit on train only - no leakage)
from sklearn.model_selection import train_test_split
trv, te = train_test_split(df, test_size=0.2, stratify=df['y'], random_state=RS)
tr, va  = train_test_split(trv, test_size=0.1, stratify=trv['y'], random_state=RS)
ytr, yva, yte = tr['y'].values, va['y'].values, te['y'].values
print('split:', len(tr), len(va), len(te))

#%% [C2] TF-IDF (1,2)-grams feature engineering (classical branch)
from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer(ngram_range=(1,2), max_features=20000, min_df=3, sublinear_tf=True)
Xtr = vec.fit_transform(tr['text']); Xva = vec.transform(va['text']); Xte = vec.transform(te['text'])

#%% [C3] imbalance handling - SVD + SMOTE (for KNN/XGB Bayes branch); class-weight for the rest
from sklearn.decomposition import TruncatedSVD
from imblearn.over_sampling import SMOTE
svd = TruncatedSVD(n_components=100, random_state=RS)
Dtr = svd.fit_transform(Xtr); Dva = svd.transform(Xva); Dte = svd.transform(Xte)
Dtr_bal, ytr_bal = SMOTE(random_state=RS).fit_resample(Dtr, ytr)

# =============================================================================
# (C4-C22: the 18 models + fusion, each model is its own cell)
# =============================================================================
from sklearn.metrics import f1_score
def macro_f1(y, pred): return round(f1_score(y, pred, average='macro'), 4)

#%% [C4] Naive Bayes (floor)
from sklearn.naive_bayes import MultinomialNB
m = MultinomialNB().fit(Xtr, ytr); print('NaiveBayes test macroF1 =', macro_f1(yte, m.predict(Xte)))

#%% [C5] Logistic Regression
from sklearn.linear_model import LogisticRegression
m = LogisticRegression(max_iter=1000, class_weight='balanced').fit(Xtr, ytr)
print('LogReg test macroF1 =', macro_f1(yte, m.predict(Xte)))

#%% [C6] LinearSVC
from sklearn.svm import LinearSVC
m = LinearSVC(class_weight='balanced').fit(Xtr, ytr); print('LinearSVC test macroF1 =', macro_f1(yte, m.predict(Xte)))

#%% [C7] Decision Tree
from sklearn.tree import DecisionTreeClassifier
m = DecisionTreeClassifier(max_depth=40, random_state=RS).fit(Dtr, ytr); print('DecisionTree test macroF1 =', macro_f1(yte, m.predict(Dte)))

#%% [C8] Random Forest
from sklearn.ensemble import RandomForestClassifier
m = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=RS).fit(Dtr, ytr); print('RandomForest test macroF1 =', macro_f1(yte, m.predict(Dte)))

#%% [C9] XGBoost
import xgboost as xgb
m = xgb.XGBClassifier(n_estimators=400, max_depth=8, tree_method='hist', n_jobs=-1).fit(Dtr, ytr)
print('XGBoost test macroF1 =', macro_f1(yte, m.predict(Dte)))

#%% [C10] KNN - BayesSearchCV + SMOTE, MCC-scored
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer, matthews_corrcoef
from skopt import BayesSearchCV
from skopt.space import Integer, Real, Categorical
opt = BayesSearchCV(KNeighborsClassifier(),
    {'n_neighbors':Integer(1,50),'leaf_size':Integer(10,100),'weights':Categorical(['uniform','distance']),'p':Integer(1,2)},
    n_iter=50, n_jobs=-1, random_state=10, scoring=make_scorer(matthews_corrcoef))
opt.fit(Dtr_bal, ytr_bal); print('KNN(Bayes) test macroF1 =', macro_f1(yte, opt.predict(Dte)))

#%% [C11] XGBoost - BayesSearchCV + SMOTE, MCC-scored
opt = BayesSearchCV(xgb.XGBClassifier(eval_metric='logloss', n_jobs=-1),
    {'n_estimators':Integer(100,800),'learning_rate':Real(0.01,0.3,'log-uniform'),'max_depth':Integer(3,12),
     'subsample':Real(0.5,1.0),'colsample_bytree':Real(0.5,1.0),'gamma':Real(0,5),
     'reg_alpha':Real(1e-8,10,'log-uniform'),'reg_lambda':Real(1e-8,10,'log-uniform')},
    n_iter=50, n_jobs=-1, random_state=10, scoring=make_scorer(matthews_corrcoef))
opt.fit(Dtr_bal, ytr_bal); print('XGB(Bayes) test macroF1 =', macro_f1(yte, opt.predict(Dte)))

#%% [C12-C16] transformer text models (fine-tuned)
# CamemBERT, FlauBERT, XLM-RoBERTa, mBERT, DistilCamemBERT - one per run (GPU).
# Use df['text_nostem'] (raw-ish). HuggingFace ids:
TRANSFORMERS = {'CamemBERT':'camembert-base','FlauBERT':'flaubert/flaubert_base_cased',
                'XLM-RoBERTa':'xlm-roberta-base','mBERT':'bert-base-multilingual-cased',
                'DistilCamemBERT':'cmarkea/distilcamembert-base'}
print('transformer models to fine-tune:', list(TRANSFORMERS))

#%% [C17-C21] image models (fine-tuned, timm)
IMAGES = {'EfficientNet-B0':'efficientnet_b0','ResNet50':'resnet50','EfficientNet-B3':'efficientnet_b3',
          'ConvNeXt-tiny':'convnext_tiny','ViT-small':'vit_small_patch16_224'}
print('image models to fine-tune:', list(IMAGES))

#%% [C22] FUSION - best val text + best val image, late fusion weight tuned on val
# w* = argmax_w macroF1(va,  w*text_val_proba + (1-w)*image_val_proba); report on test.
print('Fusion = best validation text model + best validation image model (weight tuned on val).')
