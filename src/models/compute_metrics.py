# =============================================================================
# Build a complete per-model metrics table (accuracy + macro-F1 + weighted-F1) for the report.
# - Classical: refit each model exactly as in benchmark_all_models.py (same TF-IDF/SVD/seeds) and
#   evaluate on the 2-way test set -> accuracy/macro-F1/weighted-F1.
# - Transformers + image models + fusion: load cached test-probability .npy and score them.
# - KNN/XGB-Bayes: macro-F1 carried from benchmark_results.json (search-selected test score);
#   accuracy computed if a cached prediction exists, else marked None.
# Output: reports/metrics_table.json   Run: python3 src/models/compute_metrics.py
# =============================================================================
import os, json, glob, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import f1_score, accuracy_score
RS=42; DATA='data'; CACHE='.tmp/rakuten_cache'
def m3(y,p): return {'accuracy':round(float(accuracy_score(y,p)),4),
                     'macro_f1':round(float(f1_score(y,p,average='macro')),4),
                     'weighted_f1':round(float(f1_score(y,p,average='weighted')),4)}

df=pd.read_csv(f'{DATA}/df_clean.csv'); df['text']=df['text'].fillna('')
tr,te=train_test_split(df,test_size=0.2,stratify=df['y'],random_state=RS)
ytr,yte=tr['y'].values,te['y'].values
print('split tr',len(tr),'te',len(te),flush=True)
vec=TfidfVectorizer(ngram_range=(1,2),max_features=20000,min_df=3,sublinear_tf=True)
Xtr=vec.fit_transform(tr['text']); Xte=vec.transform(te['text'])
svd=TruncatedSVD(n_components=300,random_state=RS); Dtr=svd.fit_transform(Xtr); Dte=svd.transform(Xte)
M={}
# ---- classical (refit exactly as benchmark) ----
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
print('refitting classical...',flush=True)
M['NaiveBayes']=m3(yte, MultinomialNB().fit(Xtr,ytr).predict(Xte))
M['LogReg']=m3(yte, LogisticRegression(max_iter=1000,class_weight='balanced').fit(Xtr,ytr).predict(Xte))
M['LinearSVC']=m3(yte, LinearSVC(class_weight='balanced').fit(Xtr,ytr).predict(Xte))
M['DecisionTree']=m3(yte, DecisionTreeClassifier(max_depth=40,random_state=RS).fit(Dtr,ytr).predict(Dte))
M['RandomForest']=m3(yte, RandomForestClassifier(n_estimators=300,n_jobs=-1,random_state=RS).fit(Dtr,ytr).predict(Dte))
M['XGBoost']=m3(yte, xgb.XGBClassifier(n_estimators=400,max_depth=8,tree_method='hist',n_jobs=-1).fit(Dtr,ytr).predict(Dte))
# ---- GPU models from cached probs ----
gpu=json.load(open('reports/gpu_results.json')) if os.path.exists('reports/gpu_results.json') else {}
trans={'CamemBERT':'tr_CamemBERT_te.npy','FlauBERT':'tr_FlauBERT_te.npy','XLM-RoBERTa':'tr_XLM-RoBERTa_te.npy',
       'mBERT':'tr_mBERT_te.npy','DistilCamemBERT':'tr_DistilCamemBERT_te.npy'}
imgs={'EfficientNet-B0':'img_efficientnet_b0_te.npy','ResNet50':'img_resnet50_te.npy','EfficientNet-B3':'img_efficientnet_b3_te.npy',
      'ConvNeXt-tiny':'img_convnext_tiny_te.npy','ViT-small':'img_vit_small_patch16_224_te.npy'}
for name,f in {**trans,**imgs}.items():
    p=f'{CACHE}/{f}'
    if os.path.exists(p): M[name]=m3(yte, np.load(p).argmax(1))
# ---- multimodal equal-weight fusion (best text + best image) from cached probs ----
mm=gpu.get('multimodal')
if isinstance(mm,dict):
    bt,bi=mm.get('best_text'),mm.get('best_image')
    pt,pi=f"{CACHE}/{trans.get(bt,'')}",f"{CACHE}/{imgs.get(bi,'')}"
    if os.path.exists(pt) and os.path.exists(pi):
        M['multimodal']=m3(yte,(0.5*np.load(pt)+0.5*np.load(pi)).argmax(1)); M['multimodal']['fusion']=f'{bt} + {bi} (equal weight)'
# ---- Bayes-tuned: macro-F1 from benchmark; accuracy if a cached pred exists ----
bench=json.load(open('reports/benchmark_results.json')) if os.path.exists('reports/benchmark_results.json') else {}
for k in ['KNN_BayesSMOTE','XGB_BayesSMOTE']:
    if k in bench and not isinstance(bench[k],dict):
        predf=f'{CACHE}/pred_{k}_te.npy'
        if os.path.exists(predf): M[k]=m3(yte,np.load(predf))
        else: M[k]={'accuracy':None,'macro_f1':round(float(bench[k]),4),'weighted_f1':None,'note':'search-selected test macro-F1; predictions not cached'}
json.dump(M,open('reports/metrics_table.json','w'),indent=2)
# sanity: computed macro-F1 vs stored
print('\nmodel | acc | macroF1 | weightedF1 | (stored macroF1)')
allstored={**bench,**{k:v for k,v in gpu.items() if not isinstance(v,dict)}}
for k,v in sorted(M.items(),key=lambda kv:-(kv[1]['macro_f1'] or 0)):
    s=allstored.get(k); print(f'{k:18s} {str(v["accuracy"]):7} {v["macro_f1"]:7} {str(v["weighted_f1"]):7}  (stored {s})')
print('\nwrote reports/metrics_table.json')
