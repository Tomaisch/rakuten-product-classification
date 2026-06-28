# =============================================================================
# PHASE E - REPORT. Assemble every figure + result + log into ONE self-contained HTML.
# Reads reports/figures/*.png (embeds base64), reports/*.json, reports/logs/*.log.
# Output: reports/rakuten_report.html  (open in any browser; shareable).
# Run after Phases A-D:  python src/make_report.py
# =============================================================================
import os, json, base64, glob, html
REP = 'reports'; FIG = f'{REP}/figures'; os.makedirs(FIG, exist_ok=True)

def b64(path):
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode()

def fig_block(prefix, title):
    imgs = sorted(glob.glob(f'{FIG}/{prefix}*.png'))
    if not imgs:
        return f'<p class="muted">No figures for {title} yet.</p>'
    out = ''
    for p in imgs:
        name = os.path.basename(p)
        out += f'<figure><img src="data:image/png;base64,{b64(p)}"/><figcaption>{html.escape(name)}</figcaption></figure>'
    return out

def json_block(path):
    if not os.path.exists(path):
        return '<p class="muted">not generated yet</p>'
    d = json.load(open(path))
    return f'<pre>{html.escape(json.dumps(d, indent=2)[:6000])}</pre>'

def log_block(name):
    p = f'{REP}/logs/{name}'
    if not os.path.exists(p):
        return ''
    txt = open(p).read()[-4000:]
    return f'<details><summary>console log ({html.escape(name)})</summary><pre>{html.escape(txt)}</pre></details>'

CSS = """
body{background:#0a0a0f;color:#e6e6f0;font-family:system-ui,Segoe UI,sans-serif;margin:0;padding:0 0 4rem}
header{background:linear-gradient(135deg,#16213e,#6c5ce7);padding:2rem 2.5rem}
h1{margin:0;font-size:1.6rem}h2{color:#a99bff;border-left:4px solid #6c5ce7;padding-left:.6rem;margin-top:2.4rem}
.wrap{max-width:1100px;margin:0 auto;padding:0 2.5rem}
figure{display:inline-block;background:#12121a;border:1px solid #26263a;border-radius:8px;padding:.6rem;margin:.5rem;vertical-align:top}
figure img{max-width:480px;display:block;border-radius:4px}
figcaption{color:#8a8aa0;font-size:.78rem;margin-top:.4rem;text-align:center}
pre{background:#12121a;border:1px solid #26263a;border-radius:8px;padding:1rem;overflow:auto;font-size:.8rem;color:#cfe}
.muted{color:#6a6a80}details{margin:.6rem 0}summary{cursor:pointer;color:#a99bff}
"""

PHASES = [
    ('Phase 0 - Download', 'Raw CSVs + images downloaded into <code>data/</code> (see <code>make_dataset.py</code>).', '', None, 'phase0.log'),
    ('Phase A - Descriptive analysis / EDA', 'Data checks, statistical tests, and EDA figures.', 'A', None, 'phaseAB.log'),
    ('Phase B - Preprocessing', 'Text cleaning + image preprocessing; output <code>data/df_clean.csv</code>.', 'B', None, None),
    ('Phase C - Modeling (build/train)', 'Dimension-reduction, t-SNE, model zoo, fusion.', 'C', f'{REP}/benchmark_results.json', 'phaseC.log'),
    ('Phase D - Model validation', 'CV, per-class, confusion, SHAP/LIME, fairness, McNemar, ablation, sanity checks.', 'D', f'{REP}/validation_results.json', 'phaseD.log'),
]

parts = [f'<!doctype html><meta charset="utf-8"><title>Rakuten pipeline report</title><style>{CSS}</style>']
parts.append('<header><div class="wrap"><h1>Rakuten 27-class - Pipeline Report</h1>'
             '<p>Auto-generated from reports/figures + reports/*.json. Every phase, every figure, every result.</p></div></header>')
parts.append('<div class="wrap">')
for title, desc, prefix, jpath, logname in PHASES:
    parts.append(f'<h2>{html.escape(title)}</h2><p>{desc}</p>')
    if prefix:
        parts.append(fig_block(prefix, title))
    if jpath:
        parts.append(f'<h3>Results</h3>{json_block(jpath)}')
    if logname:
        parts.append(log_block(logname))
parts.append('</div>')

os.makedirs(REP, exist_ok=True)
open(f'{REP}/rakuten_report.html', 'w').write('\n'.join(parts))
print('report written -> reports/rakuten_report.html  (figures embedded; open in a browser)')
