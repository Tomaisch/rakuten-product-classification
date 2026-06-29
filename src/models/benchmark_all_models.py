# =============================================================================
# PHASE C (classical) - MODELING on a strict 2-WAY split: TRAIN (80%) / TEST (20%).
# NO validation set. Stratified, seed 42. Models train on TRAIN, evaluated on TEST.
# Hyper-parameter tuning (KNN/XGB) uses cross-validation INSIDE train (not a held-out val).
# Transformers + image models + fusion live in gpu_phase.py (GPU). Classical here (CPU).
# Input: data/df_clean.csv. Output: reports/benchmark_results.json (+ C4/C5 figures).
# Run:  python3 src/models/benchmark_all_models.py
# =============================================================================

#%% [imports + load + 2-way split]
import os, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
RS = 42; DATA = 'data'; FIG = 'reports/figures'; os.makedirs(FIG, exist_ok=True); os.makedirs('reports', exist_ok=True)
df = pd.read_csv(f'{DATA}/df_clean.csv'); df['text'] = df['text'].fillna(''); df['text_nostem'] = df['text_nostem'].fillna('')
RESF = 'reports/benchmark_results.json'
RES = json.load(open(RESF)) if os.path.exists(RESF) else {}
def ck(): json.dump(RES, open(RESF, 'w'), indent=2, default=str)
def macro_f1(y, p): return round(float(f1_score(y, p, average='macro')), 4)

#%% [C1] 2-WAY split: train / test (80/20), stratified, seed 42 - NO val
tr, te = train_test_split(df, test_size=0.2, stratify=df['y'], random_state=RS)
ytr, yte = tr['y'].values, te['y'].values
print('2-way split -> train:', len(tr), '| test:', len(te))
assert not (set(tr['productid']) & set(te['productid'])), 'train/test id overlap (leakage)'

#%% [C2] TF-IDF (1,2)-grams (fit on TRAIN only)
from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer(ngram_range=(1,2), max_features=20000, min_df=3, sublinear_tf=True)
Xtr = vec.fit_transform(tr['text']); Xte = vec.transform(te['text'])
print('TF-IDF:', Xtr.shape, '->', Xte.shape)

#%% [C3] dense reduction + SMOTE on TRAIN (for tree / tuned models)
from sklearn.decomposition import TruncatedSVD
from imblearn.over_sampling import SMOTE
svd = TruncatedSVD(n_components=300, random_state=RS); Dtr = svd.fit_transform(Xtr); Dte = svd.transform(Xte)
Dtr_bal, ytr_bal = SMOTE(random_state=RS).fit_resample(Dtr, ytr)
print('SVD-300 var:', round(svd.explained_variance_ratio_.sum(),3), '| SMOTE rows:', Dtr_bal.shape[0])

#%% [C4] dimension-reduction comparison (full TF-IDF vs SVD/chi2/MI/NMF -> LogReg) + figure
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.decomposition import NMF
def _lr(A, B): return macro_f1(yte, LogisticRegression(max_iter=1000, solver='lbfgs').fit(A, ytr).predict(B))
dimres = [('full TF-IDF 20k', _lr(Xtr, Xte))]
for nc in [300, 1000]:
    sv = TruncatedSVD(n_components=nc, random_state=RS); dimres.append((f'SVD-{nc}', _lr(sv.fit_transform(Xtr), sv.transform(Xte))))
for k in [2000, 5000]:
    sk = SelectKBest(chi2, k=k).fit(Xtr, ytr); dimres.append((f'chi2-{k}', _lr(sk.transform(Xtr), sk.transform(Xte))))
sk = SelectKBest(mutual_info_classif, k=2000).fit(Xtr, ytr); dimres.append(('mutual_info-2000', _lr(sk.transform(Xtr), sk.transform(Xte))))
nmf = NMF(n_components=300, init='nndsvda', max_iter=200, random_state=RS); dimres.append(('NMF-300', _lr(nmf.fit_transform(Xtr), nmf.transform(Xte))))
RES['dim_reduction'] = sorted(dimres, key=lambda r: -r[1]); ck(); print(RES['dim_reduction'])
plt.figure(figsize=(8,4)); nm,vl = zip(*RES['dim_reduction']); plt.barh(list(nm)[::-1], list(vl)[::-1])
plt.xlabel('test macro-F1'); plt.title('Dimension-reduction comparison'); plt.tight_layout(); plt.savefig(f'{FIG}/C4_dim_reduction.png', dpi=120); plt.close()

#%% [C5] t-SNE 2-D (SVD-50 -> t-SNE on 3000-row train sample) + figure
from sklearn.manifold import TSNE
idx = np.random.RandomState(RS).choice(Xtr.shape[0], min(3000, Xtr.shape[0]), replace=False)
emb = TSNE(n_components=2, perplexity=30, init='pca', random_state=RS).fit_transform(TruncatedSVD(n_components=50, random_state=RS).fit_transform(Xtr)[idx])
plt.figure(figsize=(7,6)); plt.scatter(emb[:,0], emb[:,1], c=ytr[idx], cmap='tab20', s=6)
plt.title('t-SNE of TF-IDF, colored by class'); plt.tight_layout(); plt.savefig(f'{FIG}/C5_tsne.png', dpi=120); plt.close()

#%% [C6] Naive Bayes
from sklearn.naive_bayes import MultinomialNB
RES['NaiveBayes'] = macro_f1(yte, MultinomialNB().fit(Xtr, ytr).predict(Xte)); ck(); print('NaiveBayes', RES['NaiveBayes'])
#%% [C7] Logistic Regression
RES['LogReg'] = macro_f1(yte, LogisticRegression(max_iter=1000, class_weight='balanced').fit(Xtr, ytr).predict(Xte)); ck(); print('LogReg', RES['LogReg'])
#%% [C8] LinearSVC
from sklearn.svm import LinearSVC
RES['LinearSVC'] = macro_f1(yte, LinearSVC(class_weight='balanced').fit(Xtr, ytr).predict(Xte)); ck(); print('LinearSVC', RES['LinearSVC'])
#%% [C9] Decision Tree
from sklearn.tree import DecisionTreeClassifier
RES['DecisionTree'] = macro_f1(yte, DecisionTreeClassifier(max_depth=40, random_state=RS).fit(Dtr, ytr).predict(Dte)); ck(); print('DecisionTree', RES['DecisionTree'])
#%% [C10] Random Forest
from sklearn.ensemble import RandomForestClassifier
RES['RandomForest'] = macro_f1(yte, RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=RS).fit(Dtr, ytr).predict(Dte)); ck(); print('RandomForest', RES['RandomForest'])
#%% [C11] XGBoost
import xgboost as xgb
RES['XGBoost'] = macro_f1(yte, xgb.XGBClassifier(n_estimators=400, max_depth=8, tree_method='hist', n_jobs=-1).fit(Dtr, ytr).predict(Dte)); ck(); print('XGBoost', RES['XGBoost'])

#%% [C12] KNN - BayesSearchCV (3-fold CV INSIDE train) + SMOTE, MCC-scored, n_iter=15
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer, matthews_corrcoef
from skopt import BayesSearchCV
from skopt.space import Integer, Real, Categorical
if 'KNN_BayesSMOTE' not in RES:
    opt = BayesSearchCV(KNeighborsClassifier(), {'n_neighbors':Integer(1,50),'leaf_size':Integer(10,100),'weights':Categorical(['uniform','distance']),'p':Integer(1,2)},
                        n_iter=15, cv=3, n_jobs=-1, random_state=10, scoring=make_scorer(matthews_corrcoef))
    opt.fit(Dtr_bal, ytr_bal); RES['KNN_BayesSMOTE'] = macro_f1(yte, opt.predict(Dte)); ck()
print('KNN_BayesSMOTE', RES.get('KNN_BayesSMOTE'))

#%% [C13] XGBoost - BayesSearchCV (3-fold CV INSIDE train) + SMOTE, MCC-scored, n_iter=15
if 'XGB_BayesSMOTE' not in RES:
    opt = BayesSearchCV(xgb.XGBClassifier(eval_metric='logloss', n_jobs=-1),
        {'n_estimators':Integer(100,800),'learning_rate':Real(0.01,0.3,'log-uniform'),'max_depth':Integer(3,12),'subsample':Real(0.5,1.0),'colsample_bytree':Real(0.5,1.0),'gamma':Real(0,5),'reg_alpha':Real(1e-8,10,'log-uniform'),'reg_lambda':Real(1e-8,10,'log-uniform')},
        n_iter=15, cv=3, n_jobs=-1, random_state=10, scoring=make_scorer(matthews_corrcoef))
    opt.fit(Dtr_bal, ytr_bal); RES['XGB_BayesSMOTE'] = macro_f1(yte, opt.predict(Dte)); ck()
print('XGB_BayesSMOTE', RES.get('XGB_BayesSMOTE'))

#%% [save] classical results
ck(); print('classical Phase C done ->', RESF, '| transformers/images/fusion -> gpu_phase.py')
