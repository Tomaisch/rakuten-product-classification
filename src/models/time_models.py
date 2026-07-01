"""Build reports/runtime_table.json = {model: {device, seconds, note}} for the report's
Compute & runtime table. GPU deep-model times are parsed from the timestamped gpu_phase2.log;
classical CPU models are MEASURED here (fit+predict, same config as benchmark); XGB-Bayes GPU
time is read from xgb_bayes_gpu.log; retry models from their logs. Honest wall-clock only."""
import os, re, json, time, datetime, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
RS=42; RT={}
CPU='Intel i9-14900HX (CPU, 12-core)'; GPU='RTX 4070 Laptop (GPU, 8GB)'

# ---- parse GPU phase log: per-model wall = done_t - prev_done_t (sequential) ----
def parse_gpu(path):
    if not os.path.exists(path): return
    t0=None; prev=None
    for ln in open(path):
        m=re.match(r'\[(\d\d):(\d\d)\] .*?(2-way split|([\w\-]+) test macroF1)', ln)
        if not m: continue
        hh,mm=int(m.group(1)),int(m.group(2)); t=hh*60+mm
        if 'split' in ln: t0=t; prev=t; continue
        name=m.group(4)
        if name and prev is not None:
            dt=(t-prev)%(24*60)
            RT[name]={'device':GPU,'seconds':dt*60,'note':'fine-tuned 3 epochs'+(' (incl. data load)' if prev==t0 else '')}
            prev=t
parse_gpu('reports/logs/gpu_phase2.log')
if 'FlauBERT' in RT: RT['FlauBERT']['note']='fp16 run collapsed (excluded); fp32 retry timed separately'

# ---- image cache build is one-time; flag it on the first image model ----
for k in ['EfficientNet-B0']:
    if k in RT: RT[k]['note']='fine-tuned 3 epochs (incl. one-time image cache build)'

# ---- measured classical (CPU), same config as benchmark_all_models.py ----
df=pd.read_csv('data/df_clean.csv'); df['text']=df['text'].fillna('')
tr,te=train_test_split(df,test_size=0.2,stratify=df['y'],random_state=RS)
ytr=tr['y'].values
vec=TfidfVectorizer(ngram_range=(1,2),max_features=20000,min_df=3,sublinear_tf=True)
Xtr=vec.fit_transform(tr['text']); Xte=vec.transform(te['text'])
svd=TruncatedSVD(n_components=300,random_state=RS); Dtr=svd.fit_transform(Xtr); Dte=svd.transform(Xte)
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
def tfit(name, est, A, B):
    t=time.time(); est.fit(A,ytr); est.predict(B); RT[name]={'device':CPU,'seconds':round(time.time()-t,1),'note':'fit+predict'}
    print(f'{name:14s} {RT[name]["seconds"]:.1f}s', flush=True)
tfit('NaiveBayes', MultinomialNB(), Xtr, Xte)
tfit('LogReg', LogisticRegression(max_iter=1000,class_weight='balanced'), Xtr, Xte)
tfit('LinearSVC', LinearSVC(class_weight='balanced'), Xtr, Xte)
tfit('DecisionTree', DecisionTreeClassifier(max_depth=40,random_state=RS), Dtr, Dte)
tfit('RandomForest', RandomForestClassifier(n_estimators=300,n_jobs=-1,random_state=RS), Dtr, Dte)
tfit('XGBoost', xgb.XGBClassifier(n_estimators=400,max_depth=8,tree_method='hist',n_jobs=-1), Dtr, Dte)

# ---- XGB-Bayes GPU total from log ----
for ln in open('reports/logs/xgb_bayes_gpu.log') if os.path.exists('reports/logs/xgb_bayes_gpu.log') else []:
    m=re.search(r'\|\s*(\d+)s', ln)
    if m: RT['XGB_BayesSMOTE']={'device':GPU,'seconds':int(m.group(1)),'note':'BayesSearch 15 iter x 3-fold CV (n_jobs=1)'}

# ---- retry models: wall time from log file mtime - first line time (best available) ----
def file_secs(p):
    if not os.path.exists(p): return None
    try: return round(os.path.getmtime(p)-os.path.getctime(p),0)
    except Exception: return None
ds=file_secs('reports/logs/distilcamembert_retry.log')
if ds: RT['DistilCamemBERT']={'device':GPU,'seconds':ds,'note':'fp16 3 epochs (retry; CPU-throttled by concurrent job)'}
fb=file_secs('reports/logs/flaubert_retry.log')
if fb and os.path.exists('.tmp/rakuten_cache/tr_FlauBERT_te.npy'):
    RT['FlauBERT']={'device':GPU,'seconds':fb,'note':'fp32 4 epochs (retry; fixes fp16 collapse)'}

json.dump(RT, open('reports/runtime_table.json','w'), indent=2)
print('\nwrote reports/runtime_table.json with', len(RT), 'models')
