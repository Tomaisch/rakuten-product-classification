"""
PHASE C - MODELING (22 steps).
Rakuten 27-class - COMPLETE model benchmark (overnight, reproducible, no cache).

What it does (Phase C steps map exactly to README section 4):
  step 1  feature engineering (TF-IDF / raw text / image tensors)
  step 2  stratified train/val/test split, seed 42
  step 3  imbalance handling (class-weight or SMOTE; rank by macro-F1)
  steps 4-11   classical text models  (NaiveBayes, LogReg, LinearSVC, DecisionTree,
               RandomForest, XGBoost, KNN-BayesSMOTE, XGB-BayesSMOTE)
  steps 12-16  transformer text models (CamemBERT, FlauBERT, XLM-RoBERTa, mBERT, DistilCamemBERT)
  steps 17-21  image models (EfficientNet-B0, ResNet50, EfficientNet-B3, ConvNeXt-tiny, ViT-small)
  step 22  fusion (best val text + best val image, weight tuned on val, reported on test)
Also applies Phase B image steps B16-B19 (load/crop/resize/augment) in the image dataloader.
Phase 0 = make_dataset.py ; Phases A+B (text) = build_features.py.
Every justified model (see MODEL_SELECTION_JUSTIFICATION.md) trained from scratch on the
full 84,916 corpus, ONE common RS=42 split. Each model: logs wall-clock time + macro-F1,
saves val/test probabilities to probs/ (so a crash never loses finished models).
Run: /root/dli_lab_venv/bin/python benchmark_all_models.py   (nohup, overnight)
"""
import os, re, json, time, gc, traceback
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, accuracy_score
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
import warnings; warnings.filterwarnings('ignore')

RS=42
ROOT=Path('/root/AI/liora_projects/06_rakuten_multimodal'); DATA=ROOT/'data'; IMG=DATA/'images'/'image_train'
OUT=ROOT/'deliverables'/'benchmark'; (OUT/'probs').mkdir(parents=True,exist_ok=True)
t0=time.time();
def log(m): print(f'[{(time.time()-t0)/60:6.1f}m] {m}', flush=True)
RESF=OUT/'results.json'
RES=json.loads(RESF.read_text()) if RESF.exists() else {}   # resume-friendly
def save(name, val_p, test_p, secs, owner):
    np.save(OUT/'probs'/f'{name}_val.npy', val_p); np.save(OUT/'probs'/f'{name}_test.npy', test_p)
    RES[name]={'val_macro_f1':round(mf1(yva,val_p),4),'test_macro_f1':round(mf1(yte,test_p),4),
               'test_acc':round(accuracy_score(yte,test_p.argmax(1)),4),'minutes':round(secs/60,1),'owner':owner}
    RESF.write_text(json.dumps(RES,indent=2)); log(f"  [DONE] {name}: test macroF1={RES[name]['test_macro_f1']} ({RES[name]['minutes']}m, {owner})")

# ---------------- PHASE C step 2: data + common stratified split, seed 42 (full, no subsample) ----------------
import numpy as _np, random as _r; _np.random.seed(RS); _r.seed(RS)
X=pd.read_csv(DATA/'X_train_update.csv'); Y=pd.read_csv(DATA/'Y_train_CVw08PX.csv')
df=X.copy(); df['target']=Y['prdtypecode'].values
assert len(df)==84916, f'NOT full data {len(df)}'
df['designation']=df['designation'].fillna(''); df['description']=df['description'].fillna('')
classes=sorted(df['target'].unique()); df['y']=df['target'].map({c:i for i,c in enumerate(classes)})
trv,te=train_test_split(df,test_size=0.2,stratify=df['target'],random_state=RS)
tr,va=train_test_split(trv,test_size=0.1,stratify=trv['target'],random_state=RS)
yva=va['y'].values; yte=te['y'].values; ytr=tr['y'].values
def mf1(y,P): return f1_score(y,P.argmax(1),average='macro')
log(f'split train={len(tr)} val={len(va)} test={len(te)}  classes={len(classes)}')

# ---------------- PHASE B text preprocessing re-applied here (steps B1-B15, 39-step union) ----------------
TAG=re.compile(r'<[^>]+>');URL=re.compile(r'http\S+|www\.\S+');EMAIL=re.compile(r'\S+@\S+')
ENT=re.compile(r'&[a-z]+;|&#\d+;');DIM=re.compile(r'\d+\s*[xX]\s*\d+');NUM=re.compile(r'\d+')
NW=re.compile(r'[^a-zàâäéèêëïîôöùûüçœ ]');SP=re.compile(r'\s+')
STOP=set(stopwords.words('french'))|set(stopwords.words('english'))|set(stopwords.words('german'));tk=TweetTokenizer()
def clean(s):
    s=(s or '').lower();s=URL.sub(' ',s);s=EMAIL.sub(' ',s);s=TAG.sub(' ',s);s=ENT.sub(' ',s)
    s=DIM.sub(' ',s);s=NUM.sub(' ',s);s=NW.sub(' ',s);return SP.sub(' ',s).strip()
def stopstem(s): return ' '.join(w for w in tk.tokenize(s) if w not in STOP and len(w)>1)
for part in (tr,va,te):
    part['raw']=part['designation']+' '+part['description']
    part['text']=part['raw'].map(clean).map(stopstem)
log('preprocessing done')

# ================= PHASE C steps 1,3 + 4-9: CLASSICAL TEXT MODELS on TF-IDF =================
# (step 1 feature engineering = TF-IDF below; step 3 imbalance = class_weight per estimator)
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
import xgboost as xgb
vec=TfidfVectorizer(ngram_range=(1,2),max_features=20000,min_df=3,sublinear_tf=True)
Xtr=vec.fit_transform(tr['text']);Xva=vec.transform(va['text']);Xte=vec.transform(te['text'])
svd=TruncatedSVD(n_components=300,random_state=RS); Dtr=svd.fit_transform(Xtr);Dva=svd.transform(Xva);Dte=svd.transform(Xte)
def run_classical(name, est, dense, owner):
    if name in RES: return
    try:
        s=time.time(); A,Bv,Bt=(Dtr,Dva,Dte) if dense else (Xtr,Xva,Xte)
        m=CalibratedClassifierCV(est,cv=3) if not hasattr(est,'predict_proba') else est
        m.fit(A,ytr); save(name, m.predict_proba(Bv), m.predict_proba(Bt), time.time()-s, owner)
    except Exception as e: log(f'  [FAIL] {name}: {e}'); traceback.print_exc()
run_classical('NaiveBayes', MultinomialNB(), False, 'baseline')
run_classical('LogReg', LogisticRegression(max_iter=1000,class_weight='balanced'), False, 'Sandeep')
run_classical('LinearSVC', LinearSVC(class_weight='balanced'), False, 'Sandeep')
run_classical('DecisionTree', DecisionTreeClassifier(max_depth=40,random_state=RS), True, 'Sandeep')
run_classical('RandomForest', RandomForestClassifier(n_estimators=300,n_jobs=-1,random_state=RS), True, 'Sandeep')
run_classical('XGBoost', xgb.XGBClassifier(n_estimators=400,max_depth=8,tree_method='hist',n_jobs=-1), True, 'Sandeep')

# PHASE C steps 10-11: Jonathan: SMOTE + BayesSearchCV (KNN, XGBoost), MCC-scored
def run_jonathan():
    for name in ['KNN_BayesSMOTE','XGB_BayesSMOTE']:
        if name in RES: continue
        try:
            from imblearn.over_sampling import SMOTE
            from skopt import BayesSearchCV; from skopt.space import Integer, Real
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.metrics import make_scorer, matthews_corrcoef
            s=time.time()
            Xb,yb=SMOTE(random_state=RS).fit_resample(Dtr,ytr)   # SMOTE on dense SVD (tractable)
            if name.startswith('KNN'):
                est=KNeighborsClassifier(); sp={'n_neighbors':Integer(3,30),'p':Integer(1,2)}
            else:
                est=xgb.XGBClassifier(tree_method='hist',n_jobs=-1); sp={'n_estimators':Integer(200,500),'max_depth':Integer(4,10),'learning_rate':Real(0.02,0.3,'log-uniform')}
            opt=BayesSearchCV(est,sp,n_iter=25,cv=3,scoring=make_scorer(matthews_corrcoef),n_jobs=-1,random_state=RS)
            opt.fit(Xb,yb)
            best=CalibratedClassifierCV(opt.best_estimator_,cv=3).fit(Dtr,ytr) if name.startswith('KNN') else opt.best_estimator_
            pv=best.predict_proba(Dva); pt=best.predict_proba(Dte); save(name, pv, pt, time.time()-s, 'Jonathan')
        except Exception as e: log(f'  [FAIL] {name}: {e}'); traceback.print_exc()
run_jonathan()

# ================= PHASE C steps 12-16: TRANSFORMER TEXT MODELS =================
import torch
DEV='cuda' if torch.cuda.is_available() else 'cpu'
TRANSFORMERS=[  # (name, hf_id, owner)
  ('CamemBERT','camembert-base','Sandeep'),
  ('FlauBERT','flaubert/flaubert_base_cased','literature'),
  ('XLM-RoBERTa','xlm-roberta-base','Jonathan'),
  ('mBERT','bert-base-multilingual-cased','baseline'),
  ('DistilCamemBERT','cmarkea/distilcamembert-base','baseline'),
]
def run_transformer(name, hf, owner, epochs=3, maxlen=256, bs=16):
    if name in RES: return
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
        s=time.time(); torch.manual_seed(RS)
        tkz=AutoTokenizer.from_pretrained(hf)
        mdl=AutoModelForSequenceClassification.from_pretrained(hf,num_labels=len(classes)).to(DEV)
        opt=torch.optim.AdamW(mdl.parameters(),lr=2e-5)
        def enc(texts): return tkz(list(texts),truncation=True,max_length=maxlen,padding=True,return_tensors='pt')
        Xtr_t=tr['raw'].tolist(); idx=np.arange(len(Xtr_t))
        scaler=torch.cuda.amp.GradScaler()
        for ep in range(epochs):
            mdl.train(); np.random.shuffle(idx)
            for i in range(0,len(idx),bs):
                b=idx[i:i+bs]; e=enc([Xtr_t[j] for j in b]).to(DEV); y=torch.tensor(ytr[b]).to(DEV)
                opt.zero_grad()
                with torch.cuda.amp.autocast():
                    loss=mdl(**e,labels=y).loss
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            log(f'    {name} epoch {ep+1}/{epochs}')
        def probs(texts):
            mdl.eval(); out=[]
            with torch.no_grad():
                for i in range(0,len(texts),64):
                    e=enc(texts[i:i+64]).to(DEV)
                    with torch.cuda.amp.autocast(): out.append(torch.softmax(mdl(**e).logits,-1).float().cpu().numpy())
            return np.concatenate(out)
        save(name, probs(va['raw'].tolist()), probs(te['raw'].tolist()), time.time()-s, owner)
        del mdl; gc.collect(); torch.cuda.empty_cache()
    except Exception as e: log(f'  [FAIL] {name}: {e}'); traceback.print_exc(); torch.cuda.empty_cache()
for n,h,o in TRANSFORMERS: run_transformer(n,h,o)

# ================= PHASE C steps 17-21: IMAGE MODELS =================
# (the crop/resize below = PHASE B image steps B16 load, B17 autocrop, B18 resize 224, B19 augment)
import timm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
def crop(image):
    image=image.convert('RGB');a=np.array(image)
    c=(a[0,0].mean()+a[0,-1].mean()+a[-1,0].mean()+a[-1,-1].mean())/4
    rows,cols,_=np.where(a<c*0.85)
    if rows.size==0: return image.resize((224,224))
    t,b,l,r=rows.min(),rows.max(),cols.min(),cols.max()
    return image.crop((l,t,r,b)).resize((224,224)) if t!=b and l!=r else image.resize((224,224))
TF=transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
class DS(Dataset):
    def __init__(self,f,lab=True): self.f=f.reset_index(drop=True); self.lab=lab
    def __len__(self): return len(self.f)
    def __getitem__(self,i):
        r=self.f.iloc[i]; x=TF(crop(Image.open(IMG/f"image_{r['imageid']}_product_{r['productid']}.jpg")))
        return (x,int(r['y'])) if self.lab else (x,0)
IMAGES=[('EfficientNet-B0','efficientnet_b0','Thomas'),('ResNet50','resnet50','Sandeep'),
        ('EfficientNet-B3','efficientnet_b3','new'),('ConvNeXt-tiny','convnext_tiny','new'),
        ('ViT-small','vit_small_patch16_224','new')]
def run_image(name, arch, owner, epochs=8, bs=128, patience=3):
    if name in RES: return
    try:
        import torch.nn as nn
        s=time.time(); torch.manual_seed(RS)
        m=timm.create_model(arch,pretrained=True,num_classes=len(classes)).to(DEV)
        opt=torch.optim.Adam(m.parameters(),lr=1e-4); crit=nn.CrossEntropyLoss(); scaler=torch.cuda.amp.GradScaler()
        trL=DataLoader(DS(tr),batch_size=bs,shuffle=True,num_workers=8,pin_memory=True)
        vaL=DataLoader(DS(va,False),batch_size=bs,num_workers=8,pin_memory=True)
        teL=DataLoader(DS(te,False),batch_size=bs,num_workers=8,pin_memory=True)
        def infer(ld):
            m.eval(); P=[]
            with torch.no_grad():
                for x,_ in ld:
                    with torch.cuda.amp.autocast(): P.append(torch.softmax(m(x.to(DEV)),-1).float().cpu().numpy())
            return np.concatenate(P)
        best=0; beststate=None; bad=0
        for ep in range(epochs):
            m.train()
            for x,y in trL:
                opt.zero_grad()
                with torch.cuda.amp.autocast(): loss=crit(m(x.to(DEV)),y.to(DEV))
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            f=mf1(yva,infer(vaL)); log(f'    {name} epoch {ep+1}/{epochs} val={f:.4f}')
            if f>best: best=f; beststate={k:v.cpu().clone() for k,v in m.state_dict().items()}; bad=0
            else:
                bad+=1
                if bad>=patience: break
        m.load_state_dict(beststate)
        save(name, infer(vaL), infer(teL), time.time()-s, owner)
        del m; gc.collect(); torch.cuda.empty_cache()
    except Exception as e: log(f'  [FAIL] {name}: {e}'); traceback.print_exc(); torch.cuda.empty_cache()
for n,a,o in IMAGES: run_image(n,a,o)

# ================= PHASE C step 22: FUSION (best text + best image, val-tuned) =================
def best_of(family):
    cands={k:v for k,v in RES.items() if k in family and 'val_macro_f1' in v}
    return max(cands, key=lambda k: cands[k]['val_macro_f1']) if cands else None
text_models=['NaiveBayes','LogReg','LinearSVC','DecisionTree','RandomForest','XGBoost','KNN_BayesSMOTE','XGB_BayesSMOTE','CamemBERT','FlauBERT','XLM-RoBERTa','mBERT','DistilCamemBERT']
img_models=[n for n,_,_ in IMAGES]
bt, bi = best_of(text_models), best_of(img_models)
if bt and bi:
    tv=np.load(OUT/'probs'/f'{bt}_val.npy'); tt=np.load(OUT/'probs'/f'{bt}_test.npy')
    iv=np.load(OUT/'probs'/f'{bi}_val.npy'); it=np.load(OUT/'probs'/f'{bi}_test.npy')
    best=(0,None)
    for w in np.linspace(0,1,41):
        sc=mf1(yva,w*tv+(1-w)*iv)
        if sc>best[0]: best=(sc,round(float(w),3))
    w=best[1]; ft=w*tt+(1-w)*it
    RES['FUSION_best_of_all']={'text_model':bt,'image_model':bi,'val_weight_text':w,
        'val_macro_f1':round(best[0],4),'test_macro_f1':round(mf1(yte,ft),4),'test_acc':round(accuracy_score(yte,ft.argmax(1)),4)}
    RESF.write_text(json.dumps(RES,indent=2))
    log(f"FUSION best text={bt} + image={bi} (w={w}) -> TEST macroF1={RES['FUSION_best_of_all']['test_macro_f1']}")
log('ALL DONE'); print(json.dumps(RES,indent=2))
