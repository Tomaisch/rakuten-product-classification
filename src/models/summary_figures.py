"""Generate the summary figures the pipeline was missing:
  E1_all_models_metrics.png - all models ranked, macro-F1 + accuracy (grouped bar)
  E2_runtime.png            - per-model wall-clock runtime, coloured by device (CPU/GPU), log scale
Reads reports/metrics_table.json + reports/runtime_table.json. Run after compute_metrics + time_models."""
import json, os, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
FIG='reports/figures'; os.makedirs(FIG, exist_ok=True)
NICE={'KNN_BayesSMOTE':'KNN (Bayes+SMOTE)','XGB_BayesSMOTE':'XGB (Bayes+SMOTE)','multimodal':'Multimodal fusion'}
def nm(k): return NICE.get(k,k)
m=json.load(open('reports/metrics_table.json')) if os.path.exists('reports/metrics_table.json') else {}
rt=json.load(open('reports/runtime_table.json')) if os.path.exists('reports/runtime_table.json') else {}

# ---- E1: macro-F1 + accuracy, ranked ----
items=sorted(m.items(), key=lambda kv:(kv[1].get('macro_f1') or 0))
labels=[nm(k) for k,_ in items]
mf=[v.get('macro_f1') or 0 for _,v in items]
acc=[(v.get('accuracy') if v.get('accuracy') is not None else np.nan) for _,v in items]
y=np.arange(len(labels)); h=0.4
plt.figure(figsize=(9,max(5,len(labels)*0.42)))
plt.barh(y+h/2, mf, height=h, color='#6c5ce7', label='macro-F1')
plt.barh(y-h/2, acc, height=h, color='#19b39a', label='accuracy')
for i,(a,b) in enumerate(zip(mf,acc)):
    plt.text(a+.005, i+h/2, f'{a:.3f}', va='center', fontsize=7)
    if not np.isnan(b): plt.text(b+.005, i-h/2, f'{b:.3f}', va='center', fontsize=7)
plt.yticks(y, labels, fontsize=8); plt.xlabel('test score'); plt.xlim(0,1.0)
plt.title('All models - test macro-F1 and accuracy (ranked)'); plt.legend(loc='lower right'); plt.tight_layout()
plt.savefig(f'{FIG}/E1_all_models_metrics.png', dpi=130); plt.close()

# ---- E2: runtime per model, coloured by device ----
ritems=[(k,v) for k,v in rt.items() if v.get('seconds') is not None]
ritems.sort(key=lambda kv: kv[1]['seconds'])
rl=[nm(k) for k,_ in ritems]; secs=[v['seconds'] for _,v in ritems]
cols=['#19b39a' if 'GPU' in v.get('device','') else '#e08a3c' for _,v in ritems]
plt.figure(figsize=(9,max(5,len(rl)*0.42)))
yb=np.arange(len(rl)); plt.barh(yb, secs, color=cols)
for i,s in enumerate(secs):
    lab=f'{s/3600:.1f} h' if s>=3600 else (f'{s/60:.1f} m' if s>=90 else f'{s:.0f} s')
    plt.text(s*1.05, i, lab, va='center', fontsize=7)
plt.yticks(yb, rl, fontsize=8); plt.xscale('log'); plt.xlabel('wall-clock seconds (log scale)')
from matplotlib.patches import Patch
plt.legend(handles=[Patch(color='#19b39a',label='GPU (RTX 4070)'),Patch(color='#e08a3c',label='CPU (i9-14900HX)')],loc='lower right')
plt.title('Per-model training/search runtime'); plt.tight_layout()
plt.savefig(f'{FIG}/E2_runtime.png', dpi=130); plt.close()
print('wrote E1_all_models_metrics.png + E2_runtime.png')
