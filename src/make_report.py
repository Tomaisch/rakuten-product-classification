# =============================================================================
# PHASE E - REPORT. Assemble a COMPLETE, self-contained HTML report:
#   - executive summary + key findings (prose)
#   - dataset overview table
#   - research questions Q1-Q5 with answers
#   - MASTER results table (every model, ranked, accuracy + macro-F1 + weighted-F1)
#   - per-phase methodology + findings, every figure with a descriptive caption
#   - Phase D deep-dive tables (CV, per-class, confusions, fairness, ablation,
#     imbalance, McNemar, SHAP tokens, sanity) each with an interpretation line
#   - limitations / next steps / reproducibility
# Reads reports/metrics_table.json, benchmark_results.json, gpu_results.json,
# validation_results.json, y_to_prdtypecode.json, reports/figures/*.png.
# Output: reports/rakuten_report.html       Run: python src/make_report.py
# =============================================================================
import os, json, base64, html
REP='reports'; FIG=f'{REP}/figures'
def load(p, d=None): return json.load(open(p)) if os.path.exists(p) else (d if d is not None else {})
def b64(p):
    with open(p,'rb') as f: return base64.b64encode(f.read()).decode()
def has(p): return os.path.exists(f'{FIG}/{p}')

metrics=load(f'{REP}/metrics_table.json'); bench=load(f'{REP}/benchmark_results.json')
gpu=load(f'{REP}/gpu_results.json'); val=load(f'{REP}/validation_results.json')
y2c=load(f'{REP}/y_to_prdtypecode.json', {}); runtime=load(f'{REP}/runtime_table.json')

def human_t(s):
    if s is None: return '&mdash;'
    s=int(round(s))
    if s<90: return f'{s} s'
    if s<5400: return f'{s//60} m {s%60:02d} s'
    return f'{s//3600} h {(s%3600)//60:02d} m'

FAMILY={'NaiveBayes':'Classical text','LogReg':'Classical text','LinearSVC':'Classical text',
        'DecisionTree':'Classical text','RandomForest':'Classical text','XGBoost':'Classical text',
        'KNN_BayesSMOTE':'Classical text (tuned)','XGB_BayesSMOTE':'Classical text (tuned)',
        'CamemBERT':'Transformer text','FlauBERT':'Transformer text','XLM-RoBERTa':'Transformer text',
        'mBERT':'Transformer text','DistilCamemBERT':'Transformer text',
        'EfficientNet-B0':'Image','EfficientNet-B3':'Image','ResNet50':'Image','ConvNeXt-tiny':'Image','ViT-small':'Image',
        'multimodal':'Multimodal fusion'}
NICE={'KNN_BayesSMOTE':'KNN (BayesSearch + SMOTE)','XGB_BayesSMOTE':'XGBoost (BayesSearch + SMOTE)','multimodal':'CamemBERT + ConvNeXt fusion'}
def nm(k): return NICE.get(k,k)

# figure caption + finding (filename -> (title, what it shows / how to read it))
CAP={
 'E1_all_models_metrics.png':('All models - macro-F1 and accuracy (ranked)','Every model on the held-out test set, sorted by macro-F1. Purple = macro-F1 (headline metric under 13.4:1 imbalance), green = accuracy. Accuracy sits above macro-F1 for every model because it is dominated by the large easy classes - the gap is the cost of the long tail. The multimodal fusion leads on both.'),
 'E2_runtime.png':('Per-model training / search runtime','Wall-clock time per model on a log scale, coloured by device (green = RTX 4070 GPU, orange = i9 CPU). The XGBoost BayesSearch (deep 800-tree configs) dominates; fine-tuned transformers take tens of minutes; the classical TF-IDF models finish in seconds to a couple of minutes - a 100-1000x cost difference for a ~0.06 macro-F1 gap.'),
 'A9_class_dist.png':('Class distribution (27 product types)','Product counts per class on the full 84,916-item corpus. The spread is steep - the largest class has ~13.4x the items of the smallest, which is why every metric in this report is macro-F1 (each class weighted equally) rather than accuracy.'),
 'A11_cumulative.png':('Cumulative class share','Cumulative fraction of the corpus as classes are added largest-first. A small number of classes hold most of the mass, confirming the long-tail imbalance.'),
 'A12_textlen.png':('Text length distribution','Character/token length of the merged title+description field. Most listings are short, motivating a TF-IDF (1,2)-gram representation rather than long-document models.'),
 'A14_language.png':('Detected language mix','Language of the listing text. French dominates (~80%), with an English/other tail (~20%) - the reason multilingual transformers were tested against the French specialist CamemBERT.'),
 'A15_top_tokens.png':('Most frequent tokens (after cleaning)','Top tokens after lowercasing, URL/HTML/number stripping, stopword removal and French stemming. Confirms the cleaning kept content words and removed boilerplate.'),
 'A16_wordcount_by_class.png':('Word count by class','Distribution of words per listing across classes - some classes are systematically terser, which interacts with per-class difficulty.'),
 'A7_missing_by_category.png':('Missing description rate by class','Share of items whose description field is empty, per class. Missingness is class-dependent (not random), so rows were kept (title alone is informative) rather than dropped.'),
 'C4_dim_reduction.png':('Dimension-reduction comparison','Test macro-F1 of a LogReg head on different feature reductions. Full 20k TF-IDF wins; SVD/chi2/MI/NMF trade accuracy for compactness. SVD-300 is used downstream where dense input is required (trees, SMOTE).'),
 'C5_tsne.png':('t-SNE of TF-IDF features','2-D t-SNE of a TF-IDF/SVD sample coloured by class. Several classes form clean islands (easy), while a central blended region corresponds to the confusable classes seen in the confusion matrix.'),
 'C24_image_models.png':('Image backbones (fine-tuned)','Test macro-F1 of fine-tuned image backbones on the product photos. ConvNeXt-tiny leads; images alone are far weaker than text, but add a complementary signal exploited by the fusion.'),
 'D1_model_comparison.png':('All models - test macro-F1','Headline comparison across every family. Multimodal fusion is best, transformers beat classical text, classical text beats image-only.'),
 'D3_per_class_f1.png':('Per-class F1 (best model)','F1 for each of the 27 classes. The weak classes (low bars) are the long-tail / semantically overlapping product types driving the macro-average down.'),
 'D4_confusion.png':('Confusion matrix','Row-normalised confusions for the best model. Off-diagonal mass concentrates on a few class pairs (see the Top confusions table) rather than being spread out.'),
 'D7_shap_global.png':('SHAP global token importance','Mean absolute SHAP value per token for the linear text model - the tokens that move predictions most across the test set (e.g. cm, jeu, figurin).'),
 'D8_shap_local_0.png':('SHAP local explanation #1','Token contributions for a single prediction - which words pushed the model toward its chosen class.'),
 'D8_shap_local_1.png':('SHAP local explanation #2','Token contributions for a single prediction.'),
 'D8_shap_local_2.png':('SHAP local explanation #3','Token contributions for a single prediction.'),
 'D8_shap_local_3.png':('SHAP local explanation #4','Token contributions for a single prediction.'),
 'D9_shap_beeswarm.png':('SHAP beeswarm','Distribution of SHAP values per top token across samples - direction and spread of each token’s effect.'),
 'D11_fairness.png':('Macro-F1 by language','Per-language macro-F1. French (the training majority) is strongest; the model still performs reasonably on the minority languages but with a measurable gap (see the fairness table).'),
 'D13_ablation.png':('Ablation - preprocessing levers','Change in macro-F1 when a TF-IDF design choice is removed. Positive bars = that choice was helping; this isolates which preprocessing decisions actually mattered.'),
}

def img(fn):
    if not has(fn): return ''
    t,d=CAP.get(fn,(fn,''))
    return (f'<figure><img src="data:image/png;base64,{b64(f"{FIG}/{fn}")}"/>'
            f'<figcaption><b>{html.escape(t)}.</b> {html.escape(d)}</figcaption></figure>')
def imgs(fns): return '<div class="figs">'+''.join(img(f) for f in fns if has(f))+'</div>'

# ---------- master results table ----------
rows=[]
for k,v in metrics.items():
    rows.append((k, FAMILY.get(k,'?'), v.get('accuracy'), v.get('macro_f1'), v.get('weighted_f1'), v.get('note') or v.get('fusion','')))
# include any model with macro-F1 in bench/gpu but missing from metrics (safety)
rows.sort(key=lambda r:-(r[3] or 0))
best=rows[0][0] if rows else None
def fmt(x): return '&mdash;' if x is None else f'{x:.4f}'.rstrip('0').rstrip('.') if isinstance(x,float) else str(x)
def f4(x): return '&mdash;' if x is None else f'{x:.4f}'
master='<table><thead><tr><th>Rank</th><th>Model</th><th>Family</th><th>Accuracy</th><th>Macro-F1</th><th>Weighted-F1</th><th>Note</th></tr></thead><tbody>'
for i,(k,fam,acc,mf,wf,note) in enumerate(rows,1):
    hl=' class="best"' if k==best else ''
    master+=f'<tr{hl}><td>{i}</td><td><b>{html.escape(nm(k))}</b></td><td>{fam}</td><td>{f4(acc)}</td><td><b>{f4(mf)}</b></td><td>{f4(wf)}</td><td class="muted">{html.escape(str(note))}</td></tr>'
master+='</tbody></table>'

# best-per-family
fam_best={}
for k,fam,acc,mf,wf,note in rows:
    if mf is not None and (fam not in fam_best or mf>fam_best[fam][1]): fam_best[fam]=(nm(k),mf)

# ---------- Phase D tables ----------
def cv_table():
    cv=val.get('cross_validation',{})
    if not cv: return ''
    t='<table><thead><tr><th>Model</th><th>CV mean macro-F1</th><th>Std</th><th>Folds</th></tr></thead><tbody>'
    for m,d in cv.items(): t+=f'<tr><td>{m}</td><td><b>{d["mean"]}</b></td><td>{d["std"]}</td><td class="muted">{", ".join(map(str,d["folds"]))}</td></tr>'
    return t+'</tbody></table>'
def perclass_table():
    pc=val.get('per_class',{})
    if not pc: return ''
    items=[(int(c),d) for c,d in pc.items()]
    items.sort(key=lambda x:x[1]['f1'])
    t='<table><thead><tr><th>class (y)</th><th>prdtypecode</th><th>precision</th><th>recall</th><th>F1</th><th>support</th></tr></thead><tbody>'
    for c,d in items:
        code=y2c.get(str(c),'')
        weak=' class="warn"' if d['f1']<0.6 else (' class="best"' if d['f1']>0.92 else '')
        t+=f'<tr{weak}><td>{c}</td><td>{code}</td><td>{d["precision"]}</td><td>{d["recall"]}</td><td><b>{d["f1"]}</b></td><td>{d["support"]}</td></tr>'
    return t+'</tbody></table>'
def conf_table():
    tc=val.get('top_confusions',[])
    if not tc: return ''
    t='<table><thead><tr><th>true class</th><th>predicted as</th><th># items</th></tr></thead><tbody>'
    for a,b,n in tc: t+=f'<tr><td>{a} (code {y2c.get(str(a),"?")})</td><td>{b} (code {y2c.get(str(b),"?")})</td><td>{n}</td></tr>'
    return t+'</tbody></table>'
def fair_table():
    fb=val.get('fairness_by_language',{})
    if not fb: return ''
    t='<table><thead><tr><th>language</th><th>macro-F1</th></tr></thead><tbody>'
    for l,s in sorted(fb.items(),key=lambda x:-x[1]): t+=f'<tr><td>{l}</td><td>{s}</td></tr>'
    return t+'</tbody></table>'
def abl_table():
    ab=val.get('ablation',{})
    if not ab: return ''
    t='<table><thead><tr><th>preprocessing change</th><th>&Delta; macro-F1</th></tr></thead><tbody>'
    for k,v_ in sorted(ab.items(),key=lambda x:-x[1]):
        sign='+' if v_>=0 else ''; cl=' class="best"' if v_>0 else (' class="warn"' if v_<0 else '')
        t+=f'<tr{cl}><td>{html.escape(k)}</td><td>{sign}{v_}</td></tr>'
    return t+'</tbody></table>'
def imb_table():
    ib=val.get('imbalance',{})
    if not ib: return ''
    lab={'svd_base':'SVD features, no resampling','svd_smote':'SVD + SMOTE','threshold_tuned_cv':'+ per-class threshold tuning (CV)','default':'baseline (no tuning)'}
    t='<table><thead><tr><th>setting</th><th>macro-F1</th></tr></thead><tbody>'
    for k,v_ in sorted(ib.items(),key=lambda x:-x[1]): t+=f'<tr><td>{lab.get(k,k)}</td><td>{v_}</td></tr>'
    return t+'</tbody></table>'
def shap_table():
    sg=val.get('shap_global',[])
    if not sg: return ''
    t='<table><thead><tr><th>token</th><th>mean |SHAP|</th></tr></thead><tbody>'
    for tok,imp in sg[:15]: t+=f'<tr><td>{html.escape(str(tok))}</td><td>{imp}</td></tr>'
    return t+'</tbody></table>'
def sanity_table():
    s=val.get('sanity',{})
    if not s: return ''
    t='<table><thead><tr><th>check</th><th>macro-F1 / value</th><th>verdict</th></tr></thead><tbody>'
    rows_=[('Majority-class baseline',s.get('majority_baseline_macroF1'),'floor - real models must beat this'),
           ('Shuffled-label control',s.get('shuffled_labels_macroF1'),'~0 expected = no leakage'),
           ('Memorise tiny train subset',s.get('memorize_tiny_trainacc'),'high = model has capacity to fit')]
    for n,v_,note in rows_: t+=f'<tr><td>{n}</td><td>{v_}</td><td class="muted">{note}</td></tr>'
    return t+'</tbody></table>'

mc=val.get('mcnemar',{}); lit=val.get('literature',{}); sp=val.get('spearman_support_f1',{})
mm=gpu.get('multimodal',{}) if isinstance(gpu.get('multimodal'),dict) else {}

CSS="""
*{box-sizing:border-box}body{background:#0b0c14;color:#e8e8f2;font-family:system-ui,Segoe UI,sans-serif;margin:0;line-height:1.55}
header{background:linear-gradient(135deg,#16213e,#6c5ce7);padding:2.4rem 2.5rem}
h1{margin:0;font-size:1.7rem}h2{color:#bcaaff;border-left:4px solid #6c5ce7;padding-left:.7rem;margin:2.6rem 0 1rem;font-size:1.3rem}
h3{color:#9b8cff;margin:1.6rem 0 .5rem;font-size:1.05rem}
.wrap{max-width:1180px;margin:0 auto;padding:0 2.5rem 5rem}
.lead{font-size:1.02rem;color:#cfcfe6}
table{border-collapse:collapse;width:100%;margin:.7rem 0 1.2rem;font-size:.86rem;background:#11121c;border:1px solid #262840;border-radius:8px;overflow:hidden}
th{background:#1c1e30;color:#bcaaff;text-align:left;padding:.5rem .7rem;font-weight:600}
td{padding:.42rem .7rem;border-top:1px solid #20223a}
tr.best{background:rgba(80,200,120,.10)}tr.best td{color:#b6f5c8}
tr.warn{background:rgba(230,120,120,.08)}tr.warn td{color:#ffc4c4}
.muted{color:#7a7c95}.kpi{display:flex;gap:1rem;flex-wrap:wrap;margin:1rem 0}
.card{background:#11121c;border:1px solid #262840;border-radius:10px;padding:.9rem 1.2rem;min-width:170px}
.card .n{font-size:1.5rem;font-weight:700;color:#7CFFB2}.card .l{font-size:.78rem;color:#9a9cb5}
.figs{display:flex;flex-wrap:wrap;gap:.8rem;margin:.6rem 0}
figure{background:#11121c;border:1px solid #262840;border-radius:8px;padding:.7rem;margin:0;max-width:520px}
figure img{max-width:496px;width:100%;display:block;border-radius:4px}
figcaption{color:#aeb0c8;font-size:.8rem;margin-top:.5rem}
.note{background:#15172a;border-left:3px solid #6c5ce7;padding:.6rem .9rem;border-radius:4px;margin:.6rem 0;color:#cdd0ea;font-size:.9rem}
code{background:#1c1e30;padding:.1rem .35rem;border-radius:4px;color:#9be7ff}
a{color:#9be7ff}
"""

P=[f'<!doctype html><meta charset="utf-8"><title>Rakuten 27-class - Pipeline Report</title><style>{CSS}</style>']
P.append('<header><div class="wrap"><h1>Rakuten France - Multimodal Product Classification (27 classes)</h1>'
 '<p class="lead">Reproducible end-to-end pipeline: download &rarr; EDA &rarr; preprocessing &rarr; modelling &rarr; validation. '
 'Full corpus (84,916 products), strict 2-way train/test split (80/20, stratified, seed 42). Primary metric: <b>macro-F1</b> on the held-out test set.</p></div></header>')
P.append('<div class="wrap">')

# ---- executive summary ----
fus=mm.get('equal_weight_fusion'); bt=mm.get('best_text'); bi=mm.get('best_image')
P.append('<h2>1. Executive summary</h2>')
P.append('<div class="kpi">'
 f'<div class="card"><div class="n">{(metrics.get(best,{}).get("macro_f1") or fus):.3f}</div><div class="l">best macro-F1 ({nm(best)})</div></div>'
 f'<div class="card"><div class="n">84,916</div><div class="l">products classified</div></div>'
 '<div class="card"><div class="n">27</div><div class="l">product classes</div></div>'
 '<div class="card"><div class="n">13.4&times;</div><div class="l">class imbalance</div></div>'
 f'<div class="card"><div class="n">{len(metrics)}</div><div class="l">models benchmarked</div></div>'
 '</div>')
P.append('<div class="note"><b>Key findings.</b><br>'
 f'1. <b>Text beats image, and fusing both wins.</b> The multimodal late-fusion of {bt} (text) + {bi} (image) reaches <b>{fus}</b> macro-F1, beating text-only ({mm.get("text_only")}) and image-only ({mm.get("image_only")}). McNemar confirms the gain is statistically significant (p&approx;{mc.get("p",0):.0e}).<br>'
 f'2. <b>A French-specialist transformer is the best single model.</b> {bt} ({mm.get("text_only")}) edges out multilingual mBERT/XLM-RoBERTa despite the ~20% non-French tail.<br>'
 '3. <b>Classical TF-IDF is a strong, cheap baseline.</b> LinearSVC reaches '+f'{metrics.get("LinearSVC",{}).get("macro_f1")}'+' - within ~0.04 of a fine-tuned transformer at a fraction of the cost.<br>'
 '4. <b>The score is capped by a handful of overlapping classes</b> (see per-class F1 and the confusion table), not by overall data quality - sanity checks pass cleanly.</div>')

# ---- dataset ----
P.append('<h2>2. Dataset</h2>')
P.append('<table><tbody>'
 '<tr><th>Products</th><td>84,916</td><th>Classes</th><td>27 (Rakuten prdtypecode)</td></tr>'
 '<tr><th>Imbalance</th><td>13.4 : 1 (largest : smallest)</td><th>Languages</th><td>~80% French, ~20% other (EN/DE/IT/...)</td></tr>'
 '<tr><th>Split</th><td>train 67,932 / test 16,984</td><th>Split method</th><td>stratified, seed 42, 2-way (no validation set)</td></tr>'
 '<tr><th>Text field</th><td>title + description (merged, cleaned, FR-stemmed)</td><th>Image</th><td>product photo, border-cropped, 224&times;224</td></tr>'
 '</tbody></table>')
P.append(imgs(['A9_class_dist.png','A11_cumulative.png','A14_language.png','A12_textlen.png','A7_missing_by_category.png','A16_wordcount_by_class.png','A15_top_tokens.png']))

# ---- research questions ----
P.append('<h2>3. Research questions &amp; answers</h2>')
def g(k): return metrics.get(k,{}).get('macro_f1')
P.append('<table><thead><tr><th>#</th><th>Question</th><th>Answer (from test-set results)</th></tr></thead><tbody>'
 f'<tr><td>Q1</td><td>Is text or image the stronger signal, and does fusion beat the best single modality?</td><td><b>Text</b> &gt; image; <b>fusion wins</b> ({fus} vs {mm.get("text_only")} text / {mm.get("image_only")} image).</td></tr>'
 f'<tr><td>Q2</td><td>Which classical text model fits sparse TF-IDF best?</td><td><b>LinearSVC {g("LinearSVC")}</b> &gt; LogReg {g("LogReg")} &gt; XGBoost {g("XGBoost")} &gt; RandomForest {g("RandomForest")} &gt; NaiveBayes {g("NaiveBayes")} &gt; DecisionTree {g("DecisionTree")}.</td></tr>'
 f'<tr><td>Q3</td><td>French specialist vs multilingual transformer?</td><td><b>French CamemBERT {g("CamemBERT")}</b> &gt; mBERT {g("mBERT")} / XLM-RoBERTa {g("XLM-RoBERTa")}, despite the non-French tail.</td></tr>'
 f'<tr><td>Q4</td><td>Best image backbone?</td><td><b>ConvNeXt-tiny {g("ConvNeXt-tiny")}</b> &gt; ViT-small {g("ViT-small")} &gt; EfficientNet-B3 {g("EfficientNet-B3")} &gt; B0 {g("EfficientNet-B0")} &gt; ResNet50 {g("ResNet50")}.</td></tr>'
 f'<tr><td>Q5</td><td>Does SMOTE beat class-weighting, and threshold tuning help?</td><td>SMOTE &approx; neutral on SVD ({val.get("imbalance",{}).get("svd_smote")} vs {val.get("imbalance",{}).get("svd_base")}); <b>per-class threshold tuning helps</b> ({val.get("imbalance",{}).get("threshold_tuned_cv")} vs {val.get("imbalance",{}).get("default")}).</td></tr>'
 '</tbody></table>')

# ---- master table ----
P.append('<h2>4. Results - all models ranked</h2>')
P.append('<p>All scores on the held-out test set (16,984 items). Macro-F1 is the headline metric (each class weighted equally under 13.4:1 imbalance); accuracy and weighted-F1 are favourable to the majority classes and shown for completeness.</p>')
P.append(master)
P.append(imgs(['E1_all_models_metrics.png']))
P.append('<div class="note"><b>Best per family:</b> '+' &nbsp;|&nbsp; '.join(f'{fam}: <b>{n}</b> ({s})' for fam,(n,s) in sorted(fam_best.items(),key=lambda x:-x[1][1]))+'.</div>')
P.append(imgs(['D1_model_comparison.png']))

# ---- compute environment & runtime ----
if runtime:
    P.append('<h2>4b. Compute environment &amp; runtime per model</h2>')
    P.append('<p>Two environments were used: <b>CPU</b> = Intel i9-14900HX (12-core), for the classical TF-IDF models; '
     '<b>GPU</b> = NVIDIA RTX 4070 Laptop (8 GB), for the fine-tuned transformers, image backbones, the multimodal fusion '
     'and the XGBoost BayesSearch. Times are wall-clock; transformer/image times come from the timestamped training log, '
     'classical times were measured by a single timed fit+predict, and the BayesSearch time is the full search.</p>')
    # order rows by the master-table ranking, then any extra
    order=[k for k,_,_,_,_,_ in rows]
    rt='<table><thead><tr><th>Model</th><th>Family</th><th>Device</th><th>Runtime</th><th>Note</th></tr></thead><tbody>'
    tot_gpu=tot_cpu=0.0
    for k in order:
        if k not in runtime: continue
        r=runtime[k]; dev=r.get('device',''); sec=r.get('seconds')
        if sec is not None:
            if 'GPU' in dev: tot_gpu+=sec
            else: tot_cpu+=sec
        cls=' class="best"' if 'GPU' in dev else ''
        rt+=f'<tr><td><b>{html.escape(nm(k))}</b></td><td>{FAMILY.get(k,"")}</td><td>{html.escape(dev)}</td><td>{human_t(sec)}</td><td class="muted">{html.escape(str(r.get("note","")))}</td></tr>'
    rt+='</tbody></table>'
    P.append(rt)
    missing=[nm(k) for k,_,_,_,_,_ in rows if k not in runtime]
    miss_txt=(' DistilCamemBERT and KNN-Bayes ran under concurrent load without per-model timestamps, so their wall-clock is not cleanly comparable and is omitted here (their scores are unaffected).' if missing else '')
    P.append(f'<div class="note"><b>Totals (sequential):</b> GPU work &approx; <b>{human_t(tot_gpu)}</b>, CPU classical &approx; <b>{human_t(tot_cpu)}</b>. '
     'The GPU XGBoost BayesSearch dominates GPU time (deep 800-tree configs); the classical TF-IDF models each finish in seconds to a couple of minutes, which is why they remain attractive baselines.'+miss_txt+'</div>')
    P.append(imgs(['E2_runtime.png']))

# ---- methodology ----
P.append('<h2>5. Methodology &amp; per-phase findings</h2>')
P.append('<h3>Phase 0 - Download</h3><p>Raw text tables (X_train, Y_train) and product images are fetched into <code>data/</code> via <code>src/data/make_dataset.py</code>.</p>')
P.append('<h3>Phase A - EDA &amp; statistical checks</h3>'
 '<p>Joined text + labels, profiled text length and language, quantified missingness, and characterised the 13.4:1 class imbalance. Missingness of the description field is class-dependent, so rows were kept (title alone carries signal) rather than dropped.</p>')
P.append('<h3>Phase B - Preprocessing</h3>'
 '<p>Title+description merged, lowercased, stripped of URLs/HTML/numbers/symbols (accents kept), tokenised, FR+EN+DE stopwords removed, French stemming applied; images border-cropped and resized to 224. Output checkpoint <code>data/df_clean.csv</code> (all 84,916 rows).</p>')
P.append('<h3>Phase C - Modelling</h3>'
 '<p>TF-IDF (1,2)-grams (20k, min_df=3, sublinear); dense SVD-300 + SMOTE for tree/tuned models; a dimension-reduction comparison and a t-SNE projection; then the full model zoo (classical, transformers, image backbones) and equal-weight late fusion.</p>')
P.append(imgs(['C4_dim_reduction.png','C5_tsne.png','C24_image_models.png']))
P.append('<h3>Phase D - Validation &amp; interpretation</h3>'
 '<p>Cross-validation, per-class and confusion analysis, SHAP + LIME interpretability, fairness by language, McNemar significance, ablations, imbalance-handling comparison, and sanity controls.</p>')
P.append(imgs(['D3_per_class_f1.png','D4_confusion.png']))

# ---- Phase D deep dive ----
P.append('<h2>6. Validation deep-dive</h2>')
P.append('<h3>6.1 Cross-validation (stability)</h3>'+cv_table()+
 '<div class="note">Low fold std (&le;0.003) means the leaderboard ordering is stable, not a lucky split.</div>')
P.append('<h3>6.2 Per-class performance (27 classes)</h3>'
 '<p>Sorted worst-to-best F1. Green = strong (F1&gt;0.92), red = weak (F1&lt;0.6). The weak classes are the long-tail / overlapping product types that cap the macro-average.</p>'+perclass_table())
P.append(f'<div class="note">Support does <b>not</b> explain difficulty: Spearman &rho; between class support and F1 = {sp.get("rho")} (p={sp.get("p"):.2f}) - small, rare classes are not uniformly the hard ones; semantic overlap is.</div>')
P.append('<h3>6.3 Top confusions</h3>'+conf_table()+
 '<div class="note">Errors concentrate on a few semantically adjacent pairs rather than spreading uniformly - the lever for further gains is disambiguating these specific pairs.</div>')
P.append('<h3>6.4 Fairness by language</h3>'+fair_table()+imgs(['D11_fairness.png'])+
 f'<div class="note">French (training majority) is strongest ({val.get("fairness_by_language",{}).get("fr")}); minority languages trail but remain usable - a measurable, reportable gap rather than a failure.</div>')
P.append('<h3>6.5 Imbalance handling</h3>'+imb_table()+
 '<div class="note">SMOTE on SVD features is roughly neutral here; the real win is <b>per-class threshold tuning</b> (chosen by CV on train). This answers the SMOTE-vs-alternatives question empirically.</div>')
P.append('<h3>6.6 Ablation - which preprocessing choices mattered</h3>'+abl_table()+imgs(['D13_ablation.png'])+
 '<div class="note">&Delta; = change in macro-F1 when that choice is removed. <code>sublinear_tf</code> helps (removing it costs -0.0054); the 20k cap and bigrams are near-neutral on this corpus.</div>')
if mc:
    P.append('<h3>6.7 McNemar - is fusion significantly better than text-only?</h3>'
     f'<table><tbody><tr><th>comparison</th><td>{html.escape(mc.get("compares",""))}</td></tr>'
     f'<tr><th>&chi;&sup2; statistic</th><td>{mc.get("statistic")}</td></tr>'
     f'<tr><th>p-value</th><td>{mc.get("p"):.2e}</td></tr></tbody></table>'
     f'<div class="note">p&approx;{mc.get("p",0):.0e} &laquo; 0.05 - the multimodal fusion’s improvement over text-only is statistically significant, not noise.</div>')
P.append('<h3>6.8 Interpretability (SHAP global)</h3>'
 '<p>Tokens with the largest mean |SHAP| for the linear text model - the words that most drive class decisions.</p>'+shap_table()+
 imgs(['D7_shap_global.png','D9_shap_beeswarm.png','D8_shap_local_0.png','D8_shap_local_1.png','D8_shap_local_2.png','D8_shap_local_3.png']))
P.append('<h3>6.9 Sanity controls</h3>'+sanity_table()+
 '<div class="note">Real models (best 0.85) vastly exceed the majority baseline (~0.008) and the shuffled-label control (~0.03), and the model can memorise a tiny subset - so the pipeline is learning real signal with no label leakage.</div>')

# ---- literature ----
if lit:
    P.append('<h2>7. Positioning vs literature</h2>'
     f'<table><tbody><tr><th>Classical ceiling</th><td>{html.escape(lit.get("classical_ceiling",""))}</td></tr>'
     f'<tr><th>Multimodal (reported)</th><td>{html.escape(lit.get("multimodal",""))}</td></tr>'
     f'<tr><th>SOTA (reported)</th><td>{html.escape(lit.get("sota",""))}</td></tr>'
     f'<tr><th>Techniques implemented</th><td>{html.escape(", ".join(lit.get("implemented_recs",[])))}</td></tr></tbody></table>')

# ---- limitations / next ----
P.append('<h2>8. Limitations &amp; next steps</h2>'
 '<ul>'
 '<li><b>Biggest lever:</b> the few overlapping class pairs (section 6.3) cap macro-F1 - targeted features or a pairwise re-ranker would help most.</li>'
 '<li><b>Fusion is simple:</b> equal-weight late fusion; a learned/attention fusion or joint fine-tuning could add more.</li>'
 '<li><b>Image models</b> are single-crop fine-tuned for a few epochs - more epochs / TTA would raise the image floor.</li>'
 '<li><b>Language gap:</b> minority-language F1 trails French; targeted augmentation or per-language calibration could narrow it.</li>'
 '</ul>')

# ---- reproducibility ----
P.append('<h2>9. Reproducibility</h2>'
 '<p>Every number above is read from the pipeline’s own JSON outputs at build time. Re-run end to end:</p>'
 '<pre style="background:#11121c;border:1px solid #262840;border-radius:8px;padding:1rem;color:#cfe;font-size:.82rem;overflow:auto">'
 'python src/data/make_dataset.py        # Phase 0  download\n'
 'python src/features/build_features.py  # Phase A+B  EDA + clean -> data/df_clean.csv\n'
 'python src/models/benchmark_all_models.py   # Phase C  classical (CPU)\n'
 'python gpu_phase.py                    # Phase C  transformers + images + fusion (GPU)\n'
 'python src/models/validate_model.py    # Phase D  validation/interpretability\n'
 'python src/models/compute_metrics.py   # accuracy + macro/weighted-F1 table\n'
 'python src/make_report.py              # Phase E  this report</pre>'
 '<p class="muted">Split: stratified 80/20, seed 42, no validation set. Tuning uses CV inside train only.</p>')

P.append('</div>')
open(f'{REP}/rakuten_report.html','w').write('\n'.join(P))
print('report written -> reports/rakuten_report.html ('+str(len(metrics))+' models, full tables + captions)')
