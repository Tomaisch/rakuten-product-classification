"""GPU worker (run with /root/dli_lab_venv/bin/python). STRICT 2-WAY split: TRAIN(80%)/TEST(20%),
stratified seed 42, NO validation set. Trains transformers (5) + image models (5) on TRAIN,
evaluates on TEST. Multimodal = EQUAL-WEIGHT (0.5) late fusion of best text + best image (no
held-out val / no test-selection). Writes reports/gpu_results.json (resume-safe)."""
import os, json, time, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
RS=42; DATA='data'; FIG='reports/figures'; os.makedirs(FIG, exist_ok=True); os.makedirs('reports', exist_ok=True)
CACHE='.tmp/rakuten_cache'; os.makedirs(CACHE, exist_ok=True)
RESF='reports/gpu_results.json'; RES=json.load(open(RESF)) if os.path.exists(RESF) else {}
def ck(): json.dump(RES, open(RESF,'w'), indent=2, default=str)
def mf1(y,p): return round(float(f1_score(y,p,average='macro')),4)
log=lambda m: print(f'[{time.strftime("%H:%M")}] {m}', flush=True)

df=pd.read_csv(f'{DATA}/df_clean.csv'); df['text']=df['text'].fillna(''); df['text_nostem']=df['text_nostem'].fillna('')
tr,te=train_test_split(df,test_size=0.2,stratify=df['y'],random_state=RS)   # 2-WAY, no val
ytr,yte=tr['y'].values,te['y'].values
nclass=int(df['y'].nunique()); log(f'2-way split train {len(tr)} / test {len(te)} | classes {nclass}')

import torch, torch.nn as nn
from torch.amp import autocast, GradScaler
assert torch.cuda.is_available(), 'need CUDA'; DEV='cuda'; log(f'GPU: {torch.cuda.get_device_name(0)}')

# ---------------- transformers (train on TRAIN, eval on TEST) ----------------
from transformers import AutoTokenizer, AutoModelForSequenceClassification
TRANS={'CamemBERT':'camembert-base','FlauBERT':'flaubert/flaubert_base_cased','XLM-RoBERTa':'xlm-roberta-base','mBERT':'bert-base-multilingual-cased','DistilCamemBERT':'cmarkea/distilcamembert-base'}
for name,hf in TRANS.items():
    if name in RES: log(f'skip {name}'); continue
    try:
        torch.manual_seed(RS); tk=AutoTokenizer.from_pretrained(hf)
        mdl=AutoModelForSequenceClassification.from_pretrained(hf,num_labels=nclass).to(DEV)
        opt=torch.optim.AdamW(mdl.parameters(),lr=2e-5); sca=GradScaler(); Xt=tr['text_nostem'].tolist(); ii=np.arange(len(Xt))
        for ep in range(3):
            mdl.train(); np.random.shuffle(ii)
            for i in range(0,len(ii),16):
                b=ii[i:i+16]; e=tk([Xt[j] for j in b],truncation=True,max_length=256,padding=True,return_tensors='pt').to(DEV)
                y=torch.tensor(ytr[b]).to(DEV); opt.zero_grad()
                with autocast('cuda',dtype=torch.float16): loss=mdl(**e,labels=y).loss
                sca.scale(loss).backward(); sca.step(opt); sca.update()
            log(f'  {name} epoch {ep+1}/3')
        def tprobs(texts):
            mdl.eval(); out=[]
            with torch.no_grad():
                for i in range(0,len(texts),64):
                    e=tk(texts[i:i+64],truncation=True,max_length=256,padding=True,return_tensors='pt').to(DEV)
                    with autocast('cuda',dtype=torch.float16): out.append(torch.softmax(mdl(**e).logits.float(),1).cpu().numpy())
            return np.concatenate(out)
        Pte=tprobs(te['text_nostem'].tolist()); np.save(f'{CACHE}/tr_{name}_te.npy',Pte)
        RES[name]=mf1(yte,Pte.argmax(1)); ck(); log(f'{name} test macroF1 {RES[name]}')
        del mdl; torch.cuda.empty_cache()
    except Exception as e:
        log(f'FAIL {name}: {e}'); torch.cuda.empty_cache()

# ---------------- image models (train on TRAIN, eval on TEST) ----------------
import timm
from PIL import Image
IMAGES={'EfficientNet-B0':'efficientnet_b0','ResNet50':'resnet50','EfficientNet-B3':'efficientnet_b3','ConvNeXt-tiny':'convnext_tiny','ViT-small':'vit_small_patch16_224'}
N=len(df); yv=df['y'].values; tri,tei=tr.index.values,te.index.values; IMG_DIR=f'{DATA}/images/image_train'
if os.path.isdir(IMG_DIR):
    MM=f'{CACHE}/img_u8_224.mmap'
    if not os.path.exists(MM):
        log('building uint8 image cache...'); a=np.memmap(MM,dtype=np.uint8,mode='w+',shape=(N,224,224,3))
        for i,r in enumerate(df.itertuples()):
            p=f"{IMG_DIR}/image_{r.imageid}_product_{r.productid}.jpg"
            try: a[i]=np.asarray(Image.open(p).convert('RGB').resize((224,224)),dtype=np.uint8)
            except Exception: a[i]=255
            if i%10000==0: log(f'  cache {i}/{N}')
        a.flush(); del a; log('cache built')
    arr=np.memmap(MM,dtype=np.uint8,mode='r',shape=(N,224,224,3))
    MEAN=torch.tensor([0.485,0.456,0.406]).view(1,3,1,1).to(DEV); STD=torch.tensor([0.229,0.224,0.225]).view(1,3,1,1).to(DEV)
    def g(ix,train=False):
        x=torch.from_numpy(np.ascontiguousarray(arr[ix])).to(DEV).permute(0,3,1,2).float().div_(255)
        if train and torch.rand(1).item()<0.5: x=torch.flip(x,[3])
        return (x-MEAN)/STD
    def iprobs(model,idx,bs=128):
        model.eval(); P=np.zeros((len(idx),nclass),np.float32); o=np.argsort(idx); s=idx[o]; out=[]
        with torch.no_grad():
            for i in range(0,len(s),bs):
                with autocast('cuda',dtype=torch.float16): z=model(g(s[i:i+bs]))
                out.append(torch.softmax(z.float(),1).cpu().numpy())
        P[o]=np.vstack(out); return P
    BS,EP=48,3
    for disp,arch in IMAGES.items():
        if disp in RES: log(f'skip {disp}'); continue
        try:
            m=timm.create_model(arch,pretrained=True,num_classes=nclass).to(DEV)
            opt=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=1e-4); sca=GradScaler(); lf=nn.CrossEntropyLoss()
            sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=int(np.ceil(len(tri)/BS))*EP)
            for ep in range(EP):
                m.train(); perm=np.random.RandomState(ep).permutation(tri)
                for bi in range(0,len(perm),BS):
                    b=np.sort(perm[bi:bi+BS]); xb=g(b,True); yb=torch.tensor(yv[b]).to(DEV); opt.zero_grad()
                    with autocast('cuda',dtype=torch.float16): loss=lf(m(xb),yb)
                    sca.scale(loss).backward(); sca.step(opt); sca.update(); sch.step()
                log(f'  {disp} epoch {ep+1}/{EP}')
            Pte=iprobs(m,tei); np.save(f'{CACHE}/img_{arch}_te.npy',Pte)
            RES[disp]=mf1(yv[tei],Pte.argmax(1)); ck(); log(f'{disp} test macroF1 {RES[disp]}')
            del m; torch.cuda.empty_cache()
        except Exception as e:
            log(f'FAIL {disp}: {e}'); torch.cuda.empty_cache()
    iss={k:RES[k] for k in IMAGES if k in RES}
    if iss:
        od=sorted(iss.items(),key=lambda kv:-kv[1]); plt.figure(figsize=(7,4))
        plt.barh([k for k,_ in od][::-1],[v for _,v in od][::-1]); plt.xlabel('test macro-F1'); plt.title('Image models (fine-tuned, GPU)')
        plt.tight_layout(); plt.savefig(f'{FIG}/C24_image_models.png',dpi=120); plt.close()
else:
    log('no images dir - skipping image models')

# ---------------- multimodal: EQUAL-WEIGHT (0.5) late fusion of best text + best image ----------------
try:
    tkeys=[k for k in TRANS if k in RES]; ikeys=[k for k in IMAGES if k in RES]
    if tkeys and ikeys:
        bt=max(tkeys,key=lambda k:RES[k]); bi=max(ikeys,key=lambda k:RES[k])
        Pt=np.load(f'{CACHE}/tr_{bt}_te.npy'); Pi=np.load(f'{CACHE}/img_{IMAGES[bi]}_te.npy')
        fused=(0.5*Pt+0.5*Pi).argmax(1)
        RES['multimodal']={'best_text':bt,'text_only':RES[bt],'best_image':bi,'image_only':RES[bi],
                           'equal_weight_fusion':mf1(yte,fused)}; ck(); log(f"MULTIMODAL {RES['multimodal']}")
    else:
        log('fusion needs >=1 transformer and >=1 image model')
except Exception as e:
    log(f'fusion skipped: {e}')
log('GPU PHASE DONE'); print('DONE_GPU_PHASE')
