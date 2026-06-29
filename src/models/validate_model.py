# =============================================================================
# PHASE D - MODEL VALIDATION (16 steps). Each step its own #%% cell.
# CV, per-class metrics, confusion, interpretability (coef/SHAP/LIME), fairness,
# significance (Spearman, McNemar), ablation, sanity checks, literature.
# Input: data/df_clean.csv (+ reports/benchmark_results.json from Phase C).
# Figures -> reports/figures/. Reproducible: full corpus, seed=42 split.
# =============================================================================

#%% [setup] load clean data, same split, TF-IDF, reference LogReg (interpretable best linear model)
import os, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
RS = 42; DATA = 'data'; FIG = 'reports/figures'; os.makedirs(FIG, exist_ok=True); RES = {}
def macro_f1(y, p): return round(float(f1_score(y, p, average='macro')), 4)
df = pd.read_csv(f'{DATA}/df_clean.csv'); df['text'] = df['text'].fillna('')
tr, te = train_test_split(df, test_size=0.2, stratify=df['y'], random_state=RS)   # 2-WAY, no val
ytr, yte = tr['y'].values, te['y'].values
vec = TfidfVectorizer(ngram_range=(1,2), max_features=20000, min_df=3, sublinear_tf=True)
Xtr = vec.fit_transform(tr['text']); Xte = vec.transform(te['text'])
names = np.array(vec.get_feature_names_out())
best = LogisticRegression(max_iter=1000, solver='lbfgs').fit(Xtr, ytr)
yp = best.predict(Xte); classes = list(best.classes_)

#%% [D1] model-comparison ranking (from Phase C results) + figure
try:
    bm = json.load(open('reports/benchmark_results.json'))
    flat = {k: v for k, v in bm.items() if isinstance(v, (int, float))}
    order = sorted(flat.items(), key=lambda kv: -kv[1])
    plt.figure(figsize=(8,4)); plt.barh([k for k,_ in order][::-1], [v for _,v in order][::-1])
    plt.xlabel('test macro-F1'); plt.title('Model comparison (Phase C)'); plt.tight_layout()
    plt.savefig(f'{FIG}/D1_model_comparison.png', dpi=120); plt.close(); print('ranking:', order)
except Exception as e:
    print('run Phase C first for benchmark_results.json:', e)

#%% [D2] k-fold cross-validation (StratifiedKFold 3) - mean +/- std
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RS); cv = {}
for nm, m in [('LogReg', LogisticRegression(max_iter=1000, solver='lbfgs')), ('LinearSVC', LinearSVC(C=1.0))]:
    sc = cross_val_score(m, Xtr, ytr, cv=skf, scoring='f1_macro', n_jobs=-1)
    cv[nm] = {'mean': round(float(sc.mean()), 4), 'std': round(float(sc.std()), 4), 'folds': [round(float(x),4) for x in sc]}
RES['cross_validation'] = cv; print(cv)

#%% [D3] per-class metrics (classification_report)
rep = classification_report(yte, yp, output_dict=True, zero_division=0)
RES['per_class'] = {str(c): {'precision': round(rep[str(c)]['precision'],3), 'recall': round(rep[str(c)]['recall'],3),
    'f1': round(rep[str(c)]['f1-score'],3), 'support': int(rep[str(c)]['support'])} for c in classes if str(c) in rep}
f1s = [RES['per_class'][str(c)]['f1'] for c in classes if str(c) in RES['per_class']]
plt.figure(figsize=(11,4)); plt.bar(range(len(f1s)), sorted(f1s)); plt.xlabel('class (sorted)'); plt.ylabel('F1')
plt.title('Per-class F1 (sorted)'); plt.tight_layout(); plt.savefig(f'{FIG}/D3_per_class_f1.png', dpi=120); plt.close()

#%% [D4] confusion matrix figure + top confusions
cm = confusion_matrix(yte, yp, labels=classes)
plt.figure(figsize=(10,8)); plt.imshow(cm, cmap='Blues'); plt.colorbar()
plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Confusion matrix (LogReg)')
plt.tight_layout(); plt.savefig(f'{FIG}/D4_confusion.png', dpi=120); plt.close()
cmn = cm.copy(); np.fill_diagonal(cmn, 0); pairs = []
for i in range(len(classes)):
    j = int(cmn[i].argmax())
    if cmn[i, j] > 0: pairs.append((int(classes[i]), int(classes[j]), int(cmn[i, j])))
RES['top_confusions'] = sorted(pairs, key=lambda x: -x[2])[:8]; print(RES['top_confusions'])

#%% [D5] Spearman - does class support predict F1?
from scipy.stats import spearmanr
sup = [RES['per_class'][str(c)]['support'] for c in classes if str(c) in RES['per_class']]
rho, pval = spearmanr(sup, f1s); RES['spearman_support_f1'] = {'rho': round(float(rho),3), 'p': float(pval)}
print('Spearman(support, F1):', RES['spearman_support_f1'])

#%% [D6] interpretability - LogReg coefficients (top tokens per class)
coef_top = {int(c): [str(names[j]) for j in np.argsort(best.coef_[ci])[::-1][:8]] for ci, c in enumerate(classes)}
RES['coef_top'] = coef_top; print('example:', list(coef_top.items())[:2])

#%% [D7] SHAP global importance + figure
import shap
bg = Xtr[np.random.RandomState(RS).choice(Xtr.shape[0], 400, replace=False)]
ex = shap.LinearExplainer(best, bg, feature_perturbation='interventional')
N = min(700, Xte.shape[0]); sv = np.array(ex.shap_values(Xte[:N]))
if sv.ndim == 3 and sv.shape[0] == N: sv = np.transpose(sv, (2, 0, 1))
base = np.atleast_1d(np.array(ex.expected_value))
glob = np.abs(sv).mean(axis=(0, 1)); go = np.argsort(glob)[::-1][:25]
RES['shap_global'] = [(str(names[j]), round(float(glob[j]),5)) for j in go]
plt.figure(figsize=(8,8)); plt.barh([str(names[j]) for j in go][::-1], [float(glob[j]) for j in go][::-1])
plt.xlabel('mean |SHAP|'); plt.title('SHAP global - top 25 tokens'); plt.tight_layout(); plt.savefig(f'{FIG}/D7_shap_global.png', dpi=120); plt.close()

#%% [D8] SHAP local waterfalls - 3 correct + 1 wrong + figures
correct = np.where(yp[:N] == yte[:N])[0]; wrong = np.where(yp[:N] != yte[:N])[0]
pick = [int(i) for i in correct[:3]] + ([int(wrong[0])] if len(wrong) else [])
for i in pick:
    ci = classes.index(yp[i]); s = sv[ci, i, :]; o = np.argsort(np.abs(s))[::-1][:8]
    try:
        e = shap.Explanation(values=s[o], base_values=float(base[ci]) if len(base) > ci else float(base.mean()),
                             data=np.asarray(Xte[i].todense()).ravel()[o], feature_names=[str(x) for x in names[o]])
        shap.plots.waterfall(e, show=False, max_display=8); plt.title(f'obs {i} -> {int(yp[i])}')
        plt.tight_layout(); plt.savefig(f'{FIG}/D8_shap_local_{i}.png', dpi=120, bbox_inches='tight'); plt.close()
    except Exception as ee: print('waterfall', i, ee)
print('SHAP local saved for', pick)

#%% [D9] SHAP beeswarm (one class)
try:
    ci0 = int(np.bincount(yte).argmax())
    e = shap.Explanation(values=sv[ci0, :, :], data=np.asarray(Xte[:N].todense()), feature_names=[str(x) for x in names])
    shap.plots.beeswarm(e, max_display=15, show=False); plt.title(f'SHAP beeswarm - class {int(classes[ci0])}')
    plt.tight_layout(); plt.savefig(f'{FIG}/D9_shap_beeswarm.png', dpi=120, bbox_inches='tight'); plt.close()
except Exception as ee: print('beeswarm:', ee)

#%% [D10] LIME local check (independent)
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline
pipe = make_pipeline(vec, best); lime_ex = LimeTextExplainer(class_names=[str(int(c)) for c in classes])
te_r = te.reset_index(drop=True); lime_out = []
for i in pick[:2]:
    txt = str(te_r.loc[i, 'text'])
    if not txt.strip(): continue
    exp = lime_ex.explain_instance(txt, pipe.predict_proba, num_features=8, labels=[classes.index(yp[i])])
    lime_out.append({'i': int(i), 'pred': int(yp[i]), 'weights': [(w, round(float(v),4)) for w,v in exp.as_list(label=classes.index(yp[i]))]})
RES['lime'] = lime_out; print('LIME done for', [x['i'] for x in lime_out])

#%% [D11] fairness - accuracy per detected language + figure
from langdetect import detect
langs = []
for t in te_r['text']:
    try: langs.append(detect(str(t)[:200]) if str(t).strip() else 'unknown')
    except: langs.append('unknown')
langs = np.array(langs); acc_by = {}
for lg in pd.Series(langs).value_counts().head(6).index:
    mask = langs == lg
    if mask.sum() > 20: acc_by[lg] = round(float(accuracy_score(yte[mask], yp[mask])), 3)
RES['fairness_by_language'] = acc_by
plt.figure(figsize=(7,4)); plt.bar(list(acc_by), list(acc_by.values())); plt.ylabel('accuracy'); plt.title('Fairness - accuracy per language')
plt.tight_layout(); plt.savefig(f'{FIG}/D11_fairness.png', dpi=120); plt.close(); print(acc_by)

#%% [D12] McNemar significance test - text-only vs multimodal (needs cached image feats)
from statsmodels.stats.contingency_tables import mcnemar
try:
    IMAGES={'EfficientNet-B0':'efficientnet_b0','ResNet50':'resnet50','EfficientNet-B3':'efficientnet_b3','ConvNeXt-tiny':'convnext_tiny','ViT-small':'vit_small_patch16_224'}
    g = json.load(open('reports/gpu_results.json')); mm = g['multimodal']
    Pt = np.load(f".tmp/rakuten_cache/tr_{mm['best_text']}_te.npy")   # best text-model test probs
    Pi = np.load(f".tmp/rakuten_cache/img_{IMAGES[mm['best_image']]}_te.npy")  # best image-model test probs
    yp_text = Pt.argmax(1); yp_mm = (0.5*Pt + 0.5*Pi).argmax(1)       # text-only vs equal-weight fusion
    a = (yp_text == yte); b = (yp_mm == yte)
    tb = [[int((a & b).sum()), int((a & ~b).sum())], [int((~a & b).sum()), int((~a & ~b).sum())]]
    r = mcnemar(tb, exact=False, correction=True)
    RES['mcnemar'] = {'compares': 'text-only vs equal-weight multimodal fusion', 'statistic': round(float(r.statistic),3), 'p': float(r.pvalue), 'table': tb}; print(RES['mcnemar'])
except Exception as e:
    print('McNemar needs cached image features (Phase C C24):', e)

#%% [D13] ablation study - remove one TF-IDF step at a time, measure delta macro-F1 + figure
base_f1 = macro_f1(yte, best.predict(Xte)); ab = {}
configs = {'no bigrams (unigram)': dict(ngram_range=(1,1), max_features=20000, min_df=3, sublinear_tf=True),
           'no min_df': dict(ngram_range=(1,2), max_features=20000, min_df=1, sublinear_tf=True),
           'no sublinear_tf': dict(ngram_range=(1,2), max_features=20000, min_df=3, sublinear_tf=False),
           'no 20k cap': dict(ngram_range=(1,2), min_df=3, sublinear_tf=True)}
for nm, cfg in configs.items():
    v = TfidfVectorizer(**cfg); A = v.fit_transform(tr['text']); B = v.transform(te['text'])
    ab[nm] = round(macro_f1(yte, LogisticRegression(max_iter=1000, solver='lbfgs').fit(A, ytr).predict(B)) - base_f1, 4)
RES['ablation'] = ab; print('ablation (delta vs', base_f1, '):', ab)
plt.figure(figsize=(8,4)); plt.barh(list(ab), list(ab.values())); plt.xlabel('delta macro-F1 (vs full)'); plt.title('Ablation - removing each step')
plt.tight_layout(); plt.savefig(f'{FIG}/D13_ablation.png', dpi=120); plt.close()

#%% [D14] imbalance experiments - SMOTE + per-class threshold tuning
from sklearn.decomposition import TruncatedSVD
from imblearn.over_sampling import SMOTE
svd = TruncatedSVD(n_components=300, random_state=RS); Dtr = svd.fit_transform(Xtr); Dte = svd.transform(Xte)
base_svd = macro_f1(yte, LogisticRegression(max_iter=1000).fit(Dtr, ytr).predict(Dte))
Xb, yb = SMOTE(random_state=RS).fit_resample(Dtr, ytr)
smote_f1 = macro_f1(yte, LogisticRegression(max_iter=1000).fit(Xb, yb).predict(Dte))
# per-class thresholds tuned via 3-fold CV out-of-fold probs INSIDE train (no val, no test-selection)
from sklearn.model_selection import cross_val_predict
Poof = cross_val_predict(LogisticRegression(max_iter=1000, solver='lbfgs'), Xtr, ytr, cv=3, method='predict_proba', n_jobs=-1)
Pte = best.predict_proba(Xte); thr = np.ones(len(classes))
for k in range(len(classes)):
    grid = np.linspace(0.1, 0.9, 17); f1k = [f1_score(ytr == classes[k], (Poof[:,k] >= t).astype(int), zero_division=0) for t in grid]
    thr[k] = grid[int(np.argmax(f1k))]
tuned = macro_f1(yte, np.array([classes[i] for i in (Pte/thr).argmax(1)]))
RES['imbalance'] = {'svd_base': base_svd, 'svd_smote': smote_f1, 'threshold_tuned_cv': tuned, 'default': macro_f1(yte, yp)}; print(RES['imbalance'])

#%% [D15] debugging sanity checks - majority baseline, shuffle-labels, memorize-tiny, inspect predictions
from sklearn.dummy import DummyClassifier
RES['sanity'] = {}
RES['sanity']['majority_baseline_macroF1'] = macro_f1(yte, DummyClassifier(strategy='most_frequent').fit(Xtr, ytr).predict(Xte))
yshuf = np.random.RandomState(RS).permutation(ytr)
RES['sanity']['shuffled_labels_macroF1'] = macro_f1(yte, LogisticRegression(max_iter=1000).fit(Xtr, yshuf).predict(Xte))
small = LogisticRegression(max_iter=2000).fit(Xtr[:500], ytr[:500])
RES['sanity']['memorize_tiny_trainacc'] = round(float(accuracy_score(ytr[:500], small.predict(Xtr[:500]))), 3)
RES['sanity']['sample_predictions'] = [(int(yte[i]), int(yp[i])) for i in range(5)]
print('SANITY:', RES['sanity'])
# expect: majority ~0.008, shuffled ~near-0, memorize-tiny ~1.0

#%% [D16] literature / SOTA + recommended methods (implemented)
RES['literature'] = {
  'classical_ceiling': 'TF-IDF + LinearSVC ~0.75-0.79 macro-F1',
  'multimodal': 'frozen ResNet18 autocrop fusion ~0.80',
  'sota': 'fine-tuned CamemBERT/FlauBERT ~0.88-0.90 (biggest lever)',
  'implemented_recs': ['SMOTE', 'per-class threshold tuning', 'weighted late fusion', 'CamemBERT frozen embeddings']}
print(json.dumps(RES['literature'], indent=2))

#%% [save] Phase D results
os.makedirs('reports', exist_ok=True)
json.dump(RES, open('reports/validation_results.json', 'w'), indent=2, default=str)
print('Phase D done -> reports/validation_results.json ; figures -> reports/figures/')
