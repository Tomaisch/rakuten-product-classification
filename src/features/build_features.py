"""
PHASE A - DESCRIPTIVE ANALYSIS (16 steps, A1-A16)  +  PHASE B - PREPROCESSING (text, B1-B15).
src/features/build_features.py - FULL descriptive analysis + pre-processing.

What it does:
  Phase A (steps A1-A16): descriptive analysis / EDA with statistical tests + figures.
  Phase B (steps B1-B15): full TEXT preprocessing including stemming. The Phase B IMAGE
  steps (B16 load image, B17 autocrop, B18 resize 224, B19 augmentation) live in
  benchmark_all_models.py (image dataloader), because they are applied at model time.
Phase 0 (download) = make_dataset.py ; Phase C (modeling) = benchmark_all_models.py.
Implements EVERY agreed step (see PREPROCESSING_STEPS.md). Reproducible: full 84,916
corpus, no cache, seed 42.

Usage:  python src/features/build_features.py
Outputs: reports/figures/*.png, reports/eda_report.json, data/df_clean.csv
"""
import re, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import chisquare, kruskal, chi2_contingency
from statsmodels.stats.proportion import proportion_confint
import langid
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem.snowball import SnowballStemmer
from collections import Counter
import warnings; warnings.filterwarnings('ignore')

DATA = Path('data'); FIG = Path('reports/figures'); FIG.mkdir(parents=True, exist_ok=True)
RS = 42; RES = {}

# ---------------- load ----------------
X = pd.read_csv(DATA/'X_train_update.csv'); Y = pd.read_csv(DATA/'Y_train_CVw08PX.csv')
df = X.copy(); df['target'] = Y['prdtypecode'].values
df['designation'] = df['designation'].fillna(''); df['description'] = df['description'].fillna('')
raw = df['designation'] + ' ' + df['description']

# ===================== PHASE A - DESCRIPTIVE ANALYSIS (16 steps: A1-A16) =====================
# A1 view the data
RES['A1_preview'] = df.head(3)[['designation','description','productid','imageid','target']].astype(str).values.tolist()
# A2 structure: dtypes + cardinality
RES['A2_structure'] = [[c, str(df[c].dtype), int(df[c].nunique())] for c in df.columns]
RES['A2_shape'] = [int(df.shape[0]), int(df.shape[1])]
# A3 descriptive statistics (full quartiles)
def desc(s): return {k: round(float(v),1) for k,v in s.describe().items()}
RES['A3_descriptives'] = {'designation_chars': desc(df['designation'].str.len()),
                          'description_chars': desc(df['description'].str.len()),
                          'word_count': desc(raw.str.split().map(len))}
# A4 missing values per column
RES['A4_missing_by_col'] = {c: round(100*((df[c].astype(str).str.len()==0).mean() if df[c].dtype==object else df[c].isna().mean()),1) for c in df.columns}
# A5 missing-description rate BY CLASS (per-class table) + MNAR test
miss = (df['description'].str.len()==0)
md = df.assign(_m=miss).groupby('target')['_m'].mean().sort_values(ascending=False)
RES['A5_missing_desc_by_class_top6'] = {int(c): f'{v*100:.1f}%' for c,v in md.head(6).items()}
cont = pd.crosstab(df['target'], miss); chi2_m,p_m,_,_ = chi2_contingency(cont)
cramers_v = float(np.sqrt(chi2_m/(len(df)*(min(cont.shape)-1))))
RES['A5_MNAR'] = {'chi2': round(float(chi2_m),1), 'p': float(f'{p_m:.2e}'), 'cramers_v': round(cramers_v,3),
                  'verdict': 'MNAR: description-missingness depends on class' if p_m<0.05 else 'missing at random'}
# A6 class distribution + imbalance (figure)
vc = df['target'].value_counts().sort_values(ascending=False)
RES['A6_imbalance'] = {'ratio': round(float(vc.max()/vc.min()),1), 'largest_n': int(vc.max()), 'smallest_n': int(vc.min())}
plt.figure(figsize=(11,4)); plt.bar([str(c) for c in vc.index], vc.values, color='#2a7ab9')
plt.xticks(rotation=90, fontsize=7); plt.ylabel('count'); plt.title(f'Class distribution (27 classes, {RES["A6_imbalance"]["ratio"]}:1)')
plt.tight_layout(); plt.savefig(FIG/'A6_class_distribution.png', dpi=130); plt.close()
# A7 chi-square goodness-of-fit (uniform?)
chi2_g, p_g = chisquare(vc.values)
RES['A7_chi2_goodness_of_fit'] = {'chi2': round(float(chi2_g),1), 'p': float(f'{p_g:.2e}'), 'verdict': 'reject uniform - significantly imbalanced'}
# A8 cumulative imbalance curve (figure)
cum = np.cumsum(vc.values)/vc.values.sum()
plt.figure(figsize=(7,4)); plt.plot(range(1,len(cum)+1), cum, marker='o', color='#2a7ab9')
plt.xlabel('classes (largest->smallest)'); plt.ylabel('cumulative share'); plt.title('Cumulative class imbalance')
plt.tight_layout(); plt.savefig(FIG/'A8_cumulative_imbalance.png', dpi=130); plt.close()
RES['A8_top5_classes_share'] = round(float(cum[4]),3)
# A9 text length distribution (figure)
plt.figure(figsize=(9,4)); plt.hist(raw.str.len().clip(upper=2000), bins=60, color='#2a7ab9')
plt.xlabel('combined text length (chars, clipped 2000)'); plt.ylabel('count'); plt.title('Text length distribution')
plt.tight_layout(); plt.savefig(FIG/'A9_text_length.png', dpi=130); plt.close()
RES['A9_median_text_len'] = int(raw.str.len().median())
# A10 Kruskal-Wallis: text length across classes
H,p_kw = kruskal(*[raw[df['target']==c].str.len().values for c in vc.index])
RES['A10_kruskal_textlen'] = {'H': round(float(H),1), 'p': float(f'{p_kw:.2e}'), 'verdict': 'text length differs by class' if p_kw<0.05 else 'no difference'}
# A11 language detection + Wilson CIs (figure)
samp = raw.sample(min(5000,len(raw)), random_state=RS)
langs = pd.Series([langid.classify(t[:200])[0] for t in samp]).value_counts().head(6)
n = len(samp); RES['A11_languages'] = {}
for lang, cnt in langs.items():
    lo,hi = proportion_confint(cnt, n, method='wilson')
    RES['A11_languages'][lang] = {'pct': round(100*cnt/n,1), 'wilson_ci': [round(100*lo,1), round(100*hi,1)]}
plt.figure(figsize=(7,3)); langs.plot(kind='bar', color='#2a7ab9'); plt.title('Top languages (sample 5000)')
plt.tight_layout(); plt.savefig(FIG/'A11_languages.png', dpi=130); plt.close()
# A12 top tokens / words-by-class (figure)
allwords = Counter(' '.join(raw.sample(min(20000,len(raw)), random_state=RS)).lower().split())
RES['A12_top_tokens'] = [w for w,_ in allwords.most_common(25)]
# A13 duplicates
RES['A13_duplicate_texts'] = int(raw.duplicated().sum())
# A14 image inspection (white-border / size) - only if images present
img_dir = DATA/'images'/'image_train'
if img_dir.exists():
    from PIL import Image
    sizes=[]; white=0
    for _,r in df.sample(min(200,len(df)), random_state=RS).iterrows():
        p = img_dir/f"image_{r['imageid']}_product_{r['productid']}.jpg"
        if p.exists():
            im = Image.open(p).convert('RGB'); a=np.array(im); sizes.append(im.size)
            if (np.array(im.resize((50,50)))>240).mean() > 0.3: white+=1
    RES['A14_image'] = {'common_size': list(max(set(sizes), key=sizes.count)) if sizes else None,
                        'pct_with_large_white_area': round(100*white/max(len(sizes),1),1)}
else:
    RES['A14_image'] = 'images not downloaded (run make_dataset.py)'
# A15 target encoding map (codes -> 0..26)
classes = sorted(df['target'].unique())
RES['A15_encoding'] = {int(c): i for i,c in enumerate(classes)}
# A16 decisions taken from the EDA
RES['A16_decisions'] = [
  'Imbalance 13.4:1 -> rank by MACRO-F1, not accuracy',
  'Description missing ~35%, designation never -> MERGE the two text columns',
  'Multilingual (FR dominant) -> FR+EN+DE stopwords; keep ALL languages (test set has every language)',
  'Median text short -> TF-IDF (1,2)-grams; transformers for the deep route',
  'Only ids are numeric -> all signal is in text + image']

Path('reports').mkdir(exist_ok=True)
(Path('reports')/'eda_report.json').write_text(json.dumps(RES, indent=2, default=str))
print('descriptive analysis done -> reports/eda_report.json + reports/figures/*.png')

# ===================== PHASE B - PREPROCESSING, text steps B1-B15 (full, incl. stemming) =====================
# (Phase B image steps B16-B19 are applied in benchmark_all_models.py image dataloader)
stop_words = set(stopwords.words('french'))|set(stopwords.words('english'))|set(stopwords.words('german'))
tokenizer = TweetTokenizer(); stemmer = SnowballStemmer('french')
# B1 (merge designation+description) is done above via `raw`. B12 (keep all languages) = no row dropping.
def clean_text(t):
    t = t.lower()                                    # B2  lowercase
    t = re.sub(r'http\S+|www\S+', ' ', t)            # B3  strip URLs
    t = re.sub(r'\S+@\S+', ' ', t)                   # B4  strip emails
    t = re.sub(r'<.*?>', ' ', t)                     # B5  strip HTML tags (regex)
    t = re.sub(r'&[a-z]+;', ' ', t)                  # B6  strip HTML entities
    t = re.sub(r'\d+\s*x\s*\d+', ' ', t)             # B7  strip sizes "8 x 5"
    t = re.sub(r'\d+', ' ', t)                       # B8  strip bare numbers
    t = re.sub(r'[^a-zàâäéèêëïîôöùûüçœ ]', ' ', t)   # B9  strip special chars (keep accents)
    return re.sub(r'\s+', ' ', t).strip()            # B10 collapse whitespace
def tokenize_clean(t, stem=True):
    toks = tokenizer.tokenize(t)                                          # B11 TweetTokenizer
    toks = [w for w in toks if w not in stop_words and len(w) > 1]        # B13 stopwords FR+EN+DE, B14 len>1
    if stem: toks = [stemmer.stem(w) for w in toks]                       # B15 Snowball French stemming
    return ' '.join(toks)

# step-by-step demo on one product (so the steps are visible)
ex = str(raw.iloc[0]); demo = [('0 raw', ex)]
c = clean_text(ex); demo += [('1-10 cleaned', c)]
demo += [('11-14 tokenize+stopwords', tokenize_clean(c, stem=False))]
demo += [('15 stemmed', tokenize_clean(c, stem=True))]
RES['preprocessing_demo'] = [{'step': s, 'text': t[:200]} for s,t in demo]

df['t_clean'] = raw.map(clean_text)
df['text'] = df['t_clean'].map(lambda s: tokenize_clean(s, stem=True))   # keeps ALL rows / languages
RES['vocab_effect'] = {'avg_tokens_after_clean': round(float(df['t_clean'].str.split().map(len).mean()),1),
                       'avg_tokens_after_stopstem': round(float(df['text'].str.split().map(len).mean()),1)}
(Path('reports')/'eda_report.json').write_text(json.dumps(RES, indent=2, default=str))
df[['designation','description','productid','imageid','target','text']].to_csv(DATA/'df_clean.csv', index=False)
print(f'preprocessing done -> data/df_clean.csv ({len(df)} rows)')
