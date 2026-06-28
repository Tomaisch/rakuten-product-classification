# =============================================================================
# PHASE C - MODELING: build / train all candidate models (26 steps).
# Each step/model is its own independent #%% cell. Reproducible: full corpus, ONE seed=42 split.
# Input: data/df_clean.csv (from build_features.py). Saves: reports/benchmark_results.json,
# image-feature caches, and figures (C4 dim-reduction, C5 t-SNE) -> reports/figures/.
# Phase D (validate_model.py) consumes the split + these results.
#
# -----------------------------------------------------------------------------
# HARDWARE / HOW TO SWITCH CPU <-> GPU
# -----------------------------------------------------------------------------
#  - Classical models (C1-C13) are CPU-only (scikit-learn / xgboost hist). Any python works.
#  - Transformers (C14-C18) and image CNNs (C19-C24) need a CUDA GPU. They auto-detect via
#    torch.cuda.is_available(): if a GPU is present they run on it (.to('cuda'), VRAM freed
#    with torch.cuda.empty_cache() between models); if NOT, they SKIP cleanly (no CPU fine-tune,
#    which would take days).
#  - To RUN the GPU phases:  use a CUDA-enabled python, e.g.
#        /root/dli_lab_venv/bin/python benchmark_all_models.py      # torch 2.5.1+cu121, sees RTX 4070
#    To run CPU-only (classical + skip GPU phases):  any python with cpu torch, e.g.
#        python3 benchmark_all_models.py
#  - Resume/checkpoint: re-running skips models already in reports/benchmark_results.json, so you
#    can run classical on CPU first, then re-run on the GPU env to add the transformers.
#
# -----------------------------------------------------------------------------
# EXPECTED RUNTIME PER STEP  (full 84,916 corpus; CPU = ~12-core, GPU = RTX 4070 Laptop 8GB)
# -----------------------------------------------------------------------------
#  C1  split ............................ <5 s        (CPU)
#  C2  TF-IDF (20k, 1-2gram) ............. ~15 s       (CPU)
#  C3  SVD-300 + SMOTE .................. ~1-2 min     (CPU)
#  C4  dim-reduction comparison ......... ~30-60 min   (CPU) <- mutual_info + NMF on 84k x 20k = the slow one
#  C5  t-SNE (3000-sample) .............. ~1-2 min     (CPU)
#  C6  NaiveBayes ....................... ~5 s         (CPU)
#  C7  LogReg ........................... ~30-60 s     (CPU)
#  C8  LinearSVC ........................ ~30 s        (CPU)
#  C9  DecisionTree ..................... ~30 s        (CPU)
#  C10 RandomForest (300 trees) ......... ~2-4 min     (CPU)
#  C11 XGBoost (400 trees) .............. ~2-5 min     (CPU)
#  C12 KNN  BayesSearchCV (50 iters) .... ~30-60 min   (CPU) <- slow: KNN predict on SMOTE 62k x 300
#  C13 XGB  BayesSearchCV (50 iters) .... ~30-60 min   (CPU)
#  C14-C18 transformers (5x, 3 epochs) .. GPU ~30-60 min EACH (~2.5-5 h total) | CPU = impractical (days) -> skipped
#  C19-C23 image CNNs (5x, 8 epochs) .... GPU ~30-90 min EACH | needs data/images (images.zip) | CPU impractical -> skipped
#  C24 image feature extraction ......... GPU ~10-20 min | CPU ~hours
#  C25-C26 fusion ....................... ~1-2 min     (CPU, once image feats cached)
#  => Classical-only (CPU): ~2-3 h.  + Transformers (GPU): +2.5-5 h.  + Image (GPU+images): +3-6 h.
# =============================================================================

#%% [imports + load clean data]
import os, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
RS = 42; DATA = 'data'; FIG = 'reports/figures'; os.makedirs(FIG, exist_ok=True)
df = pd.read_csv(f'{DATA}/df_clean.csv')
df['text'] = df['text'].fillna(''); df['text_nostem'] = df['text_nostem'].fillna('')
os.makedirs('reports', exist_ok=True)
RESF = 'reports/benchmark_results.json'
RES = json.load(open(RESF)) if os.path.exists(RESF) else {}   # checkpoint: resume, skip done models
def _ckpt(): json.dump(RES, open(RESF, 'w'), indent=2, default=str)
def macro_f1(y, pred): return round(float(f1_score(y, pred, average='macro')), 4)

#%% [C1] stratified train/val/test split, seed 42 - no leakage (+ data checks)
trv, te = train_test_split(df, test_size=0.2, stratify=df['y'], random_state=RS)
tr, va  = train_test_split(trv, test_size=0.1, stratify=trv['y'], random_state=RS)
ytr, yva, yte = tr['y'].values, va['y'].values, te['y'].values
print('split shapes:', len(tr), len(va), len(te))
assert not (set(tr['productid']) & set(te['productid'])), 'train/test product-id overlap (leakage)'
ptr = tr['y'].value_counts(normalize=True).sort_index()
pva = va['y'].value_counts(normalize=True).sort_index().reindex(ptr.index).fillna(1e-9)
kl = float((ptr * np.log(ptr / pva)).sum())
print(f'zero train/test id-overlap OK | KL(train||val)={kl:.6f} (should be ~0)')

#%% [C2] TF-IDF (1,2)-grams feature engineering (fit on TRAIN only)
from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer(ngram_range=(1,2), max_features=20000, min_df=3, sublinear_tf=True)
Xtr = vec.fit_transform(tr['text']); Xva = vec.transform(va['text']); Xte = vec.transform(te['text'])
assert Xva.shape[1] == Xtr.shape[1], 'tfidf width mismatch train vs val'
print('TF-IDF shapes:', Xtr.shape, Xva.shape, Xte.shape)

#%% [C3] imbalance handling - SVD-300 (trees) + SMOTE (KNN/XGB Bayes); class-weight for linear
from sklearn.decomposition import TruncatedSVD
from imblearn.over_sampling import SMOTE
svd = TruncatedSVD(n_components=300, random_state=RS)
Dtr = svd.fit_transform(Xtr); Dva = svd.transform(Xva); Dte = svd.transform(Xte)
Dtr_bal, ytr_bal = SMOTE(random_state=RS).fit_resample(Dtr, ytr)
print('SVD-300 explained var:', round(svd.explained_variance_ratio_.sum(), 3), '| SMOTE rows:', Dtr_bal.shape[0])

#%% [C4] dimension-reduction comparison (full TF-IDF vs SVD vs chi2 vs MI vs NMF -> LogReg) + figure
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.decomposition import NMF
def _lrf1(A, B):
    m = LogisticRegression(max_iter=1000, solver='lbfgs').fit(A, ytr); return macro_f1(yte, m.predict(B))
dimres = [('full TF-IDF 20k', _lrf1(Xtr, Xte))]
for nc in [300, 1000]:
    sv = TruncatedSVD(n_components=nc, random_state=RS); dimres.append((f'SVD-{nc}', _lrf1(sv.fit_transform(Xtr), sv.transform(Xte))))
for k in [2000, 5000]:
    sk = SelectKBest(chi2, k=k).fit(Xtr, ytr); dimres.append((f'chi2-{k}', _lrf1(sk.transform(Xtr), sk.transform(Xte))))
sk = SelectKBest(mutual_info_classif, k=2000).fit(Xtr, ytr); dimres.append(('mutual_info-2000', _lrf1(sk.transform(Xtr), sk.transform(Xte))))
nmf = NMF(n_components=300, init='nndsvda', max_iter=200, random_state=RS); dimres.append(('NMF-300', _lrf1(nmf.fit_transform(Xtr), nmf.transform(Xte))))
RES['dim_reduction'] = sorted(dimres, key=lambda r: -r[1]); print(RES['dim_reduction'])
plt.figure(figsize=(8,4)); names_, vals_ = zip(*RES['dim_reduction'])
plt.barh(list(names_)[::-1], list(vals_)[::-1]); plt.xlabel('test macro-F1'); plt.title('Dimension-reduction comparison')
plt.tight_layout(); plt.savefig(f'{FIG}/C4_dim_reduction.png', dpi=120); plt.close()

#%% [C5] t-SNE 2-D visualization (SVD-50 -> t-SNE on a 3000-row sample) + figure
from sklearn.manifold import TSNE
idx = np.random.RandomState(RS).choice(Xtr.shape[0], min(3000, Xtr.shape[0]), replace=False)
emb = TSNE(n_components=2, perplexity=30, init='pca', random_state=RS).fit_transform(
    TruncatedSVD(n_components=50, random_state=RS).fit_transform(Xtr)[idx])
plt.figure(figsize=(7,6)); plt.scatter(emb[:,0], emb[:,1], c=ytr[idx], cmap='tab20', s=6)
plt.title('t-SNE of TF-IDF (SVD-50 -> t-SNE), colored by class'); plt.tight_layout(); plt.savefig(f'{FIG}/C5_tsne.png', dpi=120); plt.close()

#%% [C6] Naive Bayes (floor)
from sklearn.naive_bayes import MultinomialNB
RES['NaiveBayes'] = macro_f1(yte, MultinomialNB().fit(Xtr, ytr).predict(Xte)); print('NaiveBayes', RES['NaiveBayes']); _ckpt()

#%% [C7] Logistic Regression
RES['LogReg'] = macro_f1(yte, LogisticRegression(max_iter=1000, class_weight='balanced').fit(Xtr, ytr).predict(Xte)); print('LogReg', RES['LogReg']); _ckpt()

#%% [C8] LinearSVC
from sklearn.svm import LinearSVC
RES['LinearSVC'] = macro_f1(yte, LinearSVC(class_weight='balanced').fit(Xtr, ytr).predict(Xte)); print('LinearSVC', RES['LinearSVC']); _ckpt()

#%% [C9] Decision Tree
from sklearn.tree import DecisionTreeClassifier
RES['DecisionTree'] = macro_f1(yte, DecisionTreeClassifier(max_depth=40, random_state=RS).fit(Dtr, ytr).predict(Dte)); print('DecisionTree', RES['DecisionTree']); _ckpt()

#%% [C10] Random Forest
from sklearn.ensemble import RandomForestClassifier
RES['RandomForest'] = macro_f1(yte, RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=RS).fit(Dtr, ytr).predict(Dte)); print('RandomForest', RES['RandomForest']); _ckpt()

#%% [C11] XGBoost
import xgboost as xgb
RES['XGBoost'] = macro_f1(yte, xgb.XGBClassifier(n_estimators=400, max_depth=8, tree_method='hist', n_jobs=-1).fit(Dtr, ytr).predict(Dte)); print('XGBoost', RES['XGBoost']); _ckpt()

#%% [C12] KNN - BayesSearchCV + SMOTE, MCC-scored
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer, matthews_corrcoef
from skopt import BayesSearchCV
from skopt.space import Integer, Real, Categorical
opt = BayesSearchCV(KNeighborsClassifier(),
    {'n_neighbors':Integer(1,50),'leaf_size':Integer(10,100),'weights':Categorical(['uniform','distance']),'p':Integer(1,2)},
    n_iter=50, n_jobs=-1, random_state=10, scoring=make_scorer(matthews_corrcoef))
if 'KNN_BayesSMOTE' not in RES:
    opt.fit(Dtr_bal, ytr_bal); RES['KNN_BayesSMOTE'] = macro_f1(yte, opt.predict(Dte)); _ckpt()
print('KNN(Bayes)', RES.get('KNN_BayesSMOTE'))

#%% [C13] XGBoost - BayesSearchCV + SMOTE, MCC-scored
opt = BayesSearchCV(xgb.XGBClassifier(eval_metric='logloss', n_jobs=-1),
    {'n_estimators':Integer(100,800),'learning_rate':Real(0.01,0.3,'log-uniform'),'max_depth':Integer(3,12),
     'subsample':Real(0.5,1.0),'colsample_bytree':Real(0.5,1.0),'gamma':Real(0,5),
     'reg_alpha':Real(1e-8,10,'log-uniform'),'reg_lambda':Real(1e-8,10,'log-uniform')},
    n_iter=50, n_jobs=-1, random_state=10, scoring=make_scorer(matthews_corrcoef))
if 'XGB_BayesSMOTE' not in RES:
    opt.fit(Dtr_bal, ytr_bal); RES['XGB_BayesSMOTE'] = macro_f1(yte, opt.predict(Dte)); _ckpt()
print('XGB(Bayes)', RES.get('XGB_BayesSMOTE'))

#%% [C14-C18] transformer text models (fine-tuned, GPU) - one loop over the 5 transformers
TRANSFORMERS = {'CamemBERT':'camembert-base','FlauBERT':'flaubert/flaubert_base_cased',
                'XLM-RoBERTa':'xlm-roberta-base','mBERT':'bert-base-multilingual-cased',
                'DistilCamemBERT':'cmarkea/distilcamembert-base'}
try:
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('no CUDA GPU - transformer fine-tuning needs a GPU; skipping on CPU')
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    DEV = 'cuda'
    nclass = int(df['y'].nunique())
    for name, hf in TRANSFORMERS.items():
        torch.manual_seed(RS)
        tkz = AutoTokenizer.from_pretrained(hf)
        mdl = AutoModelForSequenceClassification.from_pretrained(hf, num_labels=nclass).to(DEV)
        opt_ = torch.optim.AdamW(mdl.parameters(), lr=2e-5)
        Xtr_t = tr['text_nostem'].tolist(); ii = np.arange(len(Xtr_t))
        for ep in range(3):
            mdl.train(); np.random.shuffle(ii)
            for i in range(0, len(ii), 16):
                b = ii[i:i+16]
                e = tkz([Xtr_t[j] for j in b], truncation=True, max_length=256, padding=True, return_tensors='pt').to(DEV)
                y = torch.tensor(ytr[b]).to(DEV); opt_.zero_grad(); mdl(**e, labels=y).loss.backward(); opt_.step()
        mdl.eval(); preds = []
        with torch.no_grad():
            for i in range(0, len(te), 64):
                e = tkz(te['text_nostem'].tolist()[i:i+64], truncation=True, max_length=256, padding=True, return_tensors='pt').to(DEV)
                preds.append(mdl(**e).logits.argmax(1).cpu().numpy())
        RES[name] = macro_f1(yte, np.concatenate(preds)); print(name, RES[name])
        del mdl; torch.cuda.empty_cache()
except Exception as e:
    print('transformers skipped (no GPU/deps):', e)

#%% [C19-C23] image models - fine-tune the 5 timm backbones end-to-end on GPU + image-only macro-F1
# (GPU only; reads data/images/image_train/; builds a one-time uint8 memmap cache. ~30-90 min/model.)
IMAGES = {'EfficientNet-B0':'efficientnet_b0','ResNet50':'resnet50','EfficientNet-B3':'efficientnet_b3',
          'ConvNeXt-tiny':'convnext_tiny','ViT-small':'vit_small_patch16_224'}
CACHE = '.tmp/rakuten_cache'; os.makedirs(CACHE, exist_ok=True)
img_probs = {}   # display name -> (val_probs_path, test_probs_path) for fusion
try:
    import torch, torch.nn as nn, timm
    from torch.amp import autocast, GradScaler
    from PIL import Image
    assert torch.cuda.is_available(), 'image phase needs a CUDA GPU - skipping'
    DEV = 'cuda'; IMG_DIR = f'{DATA}/images/image_train'; N = len(df)
    nclass = int(df['y'].nunique()); yv = df['y'].values
    tri, vai, tei = tr.index.values, va.index.values, te.index.values   # positions into df == memmap rows
    # one-time uint8 memmap cache (resize 224)
    MM = f'{CACHE}/img_u8_224.mmap'
    if not os.path.exists(MM):
        a = np.memmap(MM, dtype=np.uint8, mode='w+', shape=(N,224,224,3))
        for i, r in enumerate(df.itertuples()):
            p = f"{IMG_DIR}/image_{r.imageid}_product_{r.productid}.jpg"
            try: a[i] = np.asarray(Image.open(p).convert('RGB').resize((224,224)), dtype=np.uint8)
            except Exception: a[i] = 255
            if i % 10000 == 0: print('  img cache', i, '/', N, flush=True)
        a.flush(); del a; print('  uint8 image cache built')
    arr = np.memmap(MM, dtype=np.uint8, mode='r', shape=(N,224,224,3))
    MEAN = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1).to(DEV); STD = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1).to(DEV)
    def _gpu(ix, train=False):
        x = torch.from_numpy(np.ascontiguousarray(arr[ix])).to(DEV).permute(0,3,1,2).float().div_(255)
        if train and torch.rand(1).item() < 0.5: x = torch.flip(x, [3])
        return (x - MEAN) / STD
    def _probs(model, idx, bs=128):
        model.eval(); P = np.zeros((len(idx), nclass), np.float32); o = np.argsort(idx); s = idx[o]; out = []
        with torch.no_grad():
            for i in range(0, len(s), bs):
                with autocast('cuda', dtype=torch.float16): z = model(_gpu(s[i:i+bs]))
                out.append(torch.softmax(z.float(), 1).cpu().numpy())
        P[o] = np.vstack(out); return P
    BS, EP = 48, 3
    for disp, arch in IMAGES.items():
        if disp in RES: print('skip (cached)', disp); img_probs[disp]=arch; continue
        try:
            m = timm.create_model(arch, pretrained=True, num_classes=nclass).to(DEV)
            opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=1e-4)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(np.ceil(len(tri)/BS))*EP)
            sca = GradScaler(); lf = nn.CrossEntropyLoss()
            for ep in range(EP):
                m.train(); perm = np.random.RandomState(ep).permutation(tri)
                for bi in range(0, len(perm), BS):
                    b = np.sort(perm[bi:bi+BS]); xb = _gpu(b, True); yb = torch.tensor(yv[b]).to(DEV)
                    opt.zero_grad()
                    with autocast('cuda', dtype=torch.float16): loss = lf(m(xb), yb)
                    sca.scale(loss).backward(); sca.step(opt); sca.update(); sch.step()
                print(f'  {disp} epoch {ep+1}/{EP} done', flush=True)
            np.save(f'{CACHE}/imgval_{arch}.npy', _probs(m, vai))
            Pte_i = _probs(m, tei); np.save(f'{CACHE}/imgte_{arch}.npy', Pte_i)
            RES[disp] = round(float(f1_score(yv[tei], Pte_i.argmax(1), average='macro')), 4); _ckpt()
            img_probs[disp] = arch; print(disp, 'image-only macroF1', RES[disp], flush=True)
            del m; torch.cuda.empty_cache()
        except Exception as e:
            print('FAIL', disp, ':', e); torch.cuda.empty_cache()
except Exception as e:
    print('image phase skipped (no GPU/deps):', e)

#%% [C24] image-model comparison ranking (figure)
img_scores = {k: RES[k] for k in IMAGES if k in RES and isinstance(RES[k], (int, float))}
if img_scores:
    od = sorted(img_scores.items(), key=lambda kv: -kv[1])
    plt.figure(figsize=(7,4)); plt.barh([k for k,_ in od][::-1], [v for _,v in od][::-1])
    plt.xlabel('image-only test macro-F1'); plt.title('Image models (fine-tuned)'); plt.tight_layout()
    plt.savefig(f'{FIG}/C24_image_models.png', dpi=120); plt.close(); print('image ranking:', od)

#%% [C25-C26] multimodal - weighted late fusion (text LogReg + best image model), w tuned on validation
try:
    text_clf = LogisticRegression(max_iter=1000, solver='lbfgs').fit(Xtr, ytr)
    Pte_text = text_clf.predict_proba(Xte); Pva_text = text_clf.predict_proba(Xva)
    if img_scores:
        best_img = max(img_scores, key=img_scores.get); arch = IMAGES[best_img]
        Pva_img = np.load(f'{CACHE}/imgval_{arch}.npy'); Pte_img = np.load(f'{CACHE}/imgte_{arch}.npy')
        bw, bf = 0.0, -1
        for w in np.linspace(0, 1, 11):
            f = f1_score(yva, text_clf.classes_[((1-w)*Pva_text + w*Pva_img).argmax(1)], average='macro')
            if f > bf: bf, bw = f, w
        p = text_clf.classes_[((1-bw)*Pte_text + bw*Pte_img).argmax(1)]
        RES['multimodal'] = {'text_only': macro_f1(yte, Pte_text.argmax(1)), 'best_image_model': best_img,
                             'image_only': img_scores[best_img], 'weighted_fusion': macro_f1(yte, p),
                             'image_weight': round(float(bw), 2)}
    else:
        RES['multimodal'] = {'text_only': macro_f1(yte, Pte_text.argmax(1)), 'note': 'no image models trained (GPU/images needed)'}
    _ckpt(); print('MULTIMODAL:', RES['multimodal'])
except Exception as e:
    print('multimodal fusion skipped:', e)

#%% [save] Phase C results for Phase D
os.makedirs('reports', exist_ok=True)
json.dump(RES, open('reports/benchmark_results.json', 'w'), indent=2, default=str)
print('Phase C done -> reports/benchmark_results.json ; figures -> reports/figures/')
