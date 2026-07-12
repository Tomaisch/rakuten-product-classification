# =============================================================================
# PHASE A - DESCRIPTIVE ANALYSIS (18 steps) + PHASE B - PREPROCESSING (22 steps)
# Each step is its own independent #%% cell. No functions where avoidable.
# Steps follow the README pipeline outline. Reproducible: full 84,916 corpus, seed 42, no cache.
# Output: data/df_clean.csv ; figures -> reports/figures/
# =============================================================================

#%% [imports]
import os, re
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from langdetect import detect
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem.snowball import SnowballStemmer
from scipy.stats import chisquare, kruskal, chi2_contingency
from statsmodels.stats.proportion import proportion_confint
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from collections import Counter
RS = 42; DATA = 'data'; FIG = 'reports/figures'; os.makedirs(FIG, exist_ok=True)

# =============================================================================
# PHASE A - DESCRIPTIVE ANALYSIS / EDA
# =============================================================================
#%% [A1] inspect X and Y SEPARATELY before merge (data-check)
X = pd.read_csv(f'{DATA}/X_train_update.csv', index_col=0)
Y = pd.read_csv(f'{DATA}/Y_train_CVw08PX.csv', index_col=0)
print('X.shape', X.shape, '| Y.shape', Y.shape)
print(X.head(3)); print(Y.head(3)); print('X dtypes:', dict(X.dtypes))
assert len(X) == len(Y), 'X and Y have different row counts'
assert (X.index == Y.index).all(), 'row keys do not line up - cannot join'
print('row keys identical -> safe to join')

#%% [A2 / B1] merge X+Y, fill missing, build combined text
df = X.join(Y)
df['designation'] = df['designation'].fillna(''); df['description'] = df['description'].fillna('')
assert len(df) == 83502, 'unexpected row count after dedup'
text = df['designation'] + ' ' + df['description']   # B1 merge (single source of truth)

#%% [A3] view the data
print(df.head(3)[['designation','description','productid','imageid','prdtypecode']])

#%% [A4] structure - info(), dtypes, cardinality
df.info()
print('cardinality:', df.nunique().to_dict())

#%% [A5] descriptive stats - char + word lengths
print('designation chars:', df['designation'].str.len().describe())
print('description chars:', df['description'].str.len().describe())
print('combined words   :', text.str.split().map(len).describe())

#%% [A6] missing values per column
print(df[['designation','description']].apply(lambda c: (c.str.len()==0).sum()))

#%% [A7] missing-description rate BY CLASS + MNAR test (chi2 + Cramer's V) + FIGURE
no_desc = (df['description'].str.len()==0).astype(int)
ct = pd.crosstab(df['prdtypecode'], no_desc)
chi2, p, dof, _ = chi2_contingency(ct)
cramers_v = np.sqrt(chi2/(len(df)*(min(ct.shape)-1)))
print(f'MNAR chi2={chi2:.0f} p={p:.2e} CramersV={cramers_v:.3f}')
miss_by_class = df.assign(nd=no_desc).groupby('prdtypecode')['nd'].mean().sort_values(ascending=False)
plt.figure(figsize=(11,5)); plt.bar([str(c) for c in miss_by_class.index], miss_by_class.values*100)
plt.xticks(rotation=90, fontsize=7); plt.ylabel('% description missing'); plt.title(f'Missing-description rate by category (MNAR, Cramer V={cramers_v:.2f})')
plt.tight_layout(); plt.savefig(f'{FIG}/A7_missing_by_category.png', dpi=120); plt.close()

#%% [A8] duplicate analysis (exact + text-subset)
print('exact dup combined-text rows:', int(text.duplicated().sum()))
print('dup on (designation,description):', int(df.duplicated(subset=['designation','description']).sum()))

#%% [A9] class distribution + imbalance (figure)
vc = df['prdtypecode'].value_counts().sort_values(ascending=False)
plt.figure(figsize=(11,5)); plt.bar([str(c) for c in vc.index], vc.values)
plt.xticks(rotation=90, fontsize=7); plt.title(f'Class distribution (27, imbalance {vc.max()/vc.min():.1f}:1)')
plt.tight_layout(); plt.savefig(f'{FIG}/A9_class_dist.png', dpi=120); plt.close()
print('imbalance ratio:', round(vc.max()/vc.min(),1))

#%% [A10] chi-square goodness-of-fit (is the class balance uniform?)
print('GOF:', chisquare(vc.values))

#%% [A11] cumulative imbalance curve (figure)
cum = np.cumsum(vc.values)/vc.values.sum()
plt.figure(); plt.plot(range(1,28), cum, marker='o'); plt.xlabel('classes'); plt.ylabel('cumulative share')
plt.title('Cumulative imbalance'); plt.tight_layout(); plt.savefig(f'{FIG}/A11_cumulative.png', dpi=120); plt.close()

#%% [A12] text-length distribution (figure)
plt.figure(figsize=(9,4)); plt.hist(text.str.len().clip(upper=2000), bins=60)
plt.xlabel('combined length (chars, clipped 2000)'); plt.title('Text-length distribution')
plt.tight_layout(); plt.savefig(f'{FIG}/A12_textlen.png', dpi=120); plt.close()

#%% [A13] Kruskal-Wallis - text length across classes
groups = [text[df['prdtypecode']==c].str.len().values for c in vc.index]
print('Kruskal-Wallis:', kruskal(*groups))

#%% [A14] language detection + Wilson CIs (figure)
samp = text.sample(2000, random_state=RS)
detected = []
for t in samp:
    try: detected.append(detect(t[:200]))
    except: detected.append('unknown')
langs = pd.Series(detected).value_counts()
top = langs.head(6)
print((langs/len(samp)*100).head(6))
ci = [proportion_confint(n, len(samp), method='wilson') for n in top.values]
print('Wilson CIs:', [tuple(round(x,3) for x in c) for c in ci])
plt.figure(figsize=(8,4)); plt.bar(top.index, (top.values/len(samp))*100,
    yerr=[[(top.values[i]/len(samp)-ci[i][0])*100 for i in range(len(top))],
          [(ci[i][1]-top.values[i]/len(samp))*100 for i in range(len(top))]], capsize=4)
plt.ylabel('% of products'); plt.title('Language distribution (Wilson 95% CIs)')
plt.tight_layout(); plt.savefig(f'{FIG}/A14_language.png', dpi=120); plt.close()

#%% [A15] top tokens / vocabulary size (figure)
cnt = Counter(' '.join(text.sample(5000, random_state=RS)).lower().split())
print('vocab (sample):', len(cnt), '| top20:', cnt.most_common(20))
tt = cnt.most_common(20)
plt.figure(figsize=(9,5)); plt.barh([w for w,_ in tt][::-1], [n for _,n in tt][::-1])
plt.xlabel('count'); plt.title('Top-20 tokens (5k sample)'); plt.tight_layout(); plt.savefig(f'{FIG}/A15_top_tokens.png', dpi=120); plt.close()

#%% [A16] bivariate - median word count per class (figure)
wc_by_class = text.str.split().map(len).groupby(df['prdtypecode']).median().sort_values(ascending=False)
plt.figure(figsize=(11,4)); plt.bar([str(c) for c in wc_by_class.index], wc_by_class.values)
plt.xticks(rotation=90, fontsize=7); plt.ylabel('median word count'); plt.title('Median word count per class')
plt.tight_layout(); plt.savefig(f'{FIG}/A16_wordcount_by_class.png', dpi=120); plt.close()

#%% [A17] target encoding (codes -> 0..26)
le = LabelEncoder(); df['y'] = le.fit_transform(df['prdtypecode'])
print('classes:', len(le.classes_))

#%% [A18] decisions taken from the EDA
print('Decisions: merge text; keep ALL languages; rank by macro-F1; class-weight/SMOTE for imbalance.')

# =============================================================================
# PHASE B - PREPROCESSING, TEXT (B1 merged above as `text`; B2-B17 here) - KEEPS ALL ROWS
# =============================================================================
#%% [B2] lowercase
text = text.str.lower()

#%% [B3] strip URLs
text = text.str.replace(r'http\S+|www\S+', ' ', regex=True)

#%% [B4] strip emails
text = text.str.replace(r'\S+@\S+', ' ', regex=True)

#%% [B5] strip HTML tags (BeautifulSoup)
text = text.map(lambda s: BeautifulSoup(s, 'html.parser').get_text())

#%% [B6] strip HTML entities (&eacute;)
text = text.str.replace(r'&[a-z]+;', ' ', regex=True)

#%% [B7] strip sizes "8 x 5"
text = text.str.replace(r'\d+\s*x\s*\d+', ' ', regex=True)

#%% [B8] strip bare numbers
text = text.str.replace(r'\d+', ' ', regex=True)

#%% [B9] strip special chars, keep French accents
text = text.str.replace(r'[^a-zàâäéèêëïîôöùûüçœ ]', ' ', regex=True)

#%% [B10] collapse whitespace
text = text.str.replace(r'\s+', ' ', regex=True).str.strip()

#%% [B11] tokenize - TweetTokenizer
tokens = text.map(TweetTokenizer().tokenize)

#%% [B12] keep ALL languages - no row dropping (deliberate: jury/test set is multilingual)

#%% [B13] remove stopwords FR + EN + DE
STOP = set(stopwords.words('french')) | set(stopwords.words('english')) | set(stopwords.words('german'))
tokens = tokens.map(lambda ws: [w for w in ws if w not in STOP])

#%% [B14] drop length-1 tokens
tokens = tokens.map(lambda ws: [w for w in ws if len(w) > 1])

#%% [B15] Snowball-French stemming (classical branch; OFF for transformers)
stemmer = SnowballStemmer('french')
df['text']        = tokens.map(lambda ws: ' '.join(stemmer.stem(w) for w in ws))   # classical / TF-IDF
df['text_nostem'] = tokens.map(lambda ws: ' '.join(ws))                            # transformers

#%% [B16] preprocessing step-by-step demo on ONE real product (so each step is visible)
ex = str((df['designation'] + ' ' + df['description']).iloc[0])
s = ex.lower()
s = re.sub(r'http\S+|www\S+', ' ', s); s = re.sub(r'\S+@\S+', ' ', s)
s = BeautifulSoup(s, 'html.parser').get_text(); s = re.sub(r'&[a-z]+;', ' ', s)
s = re.sub(r'\d+\s*x\s*\d+', ' ', s); s = re.sub(r'\d+', ' ', s)
s = re.sub(r'[^a-zàâäéèêëïîôöùûüçœ ]', ' ', s); s = re.sub(r'\s+', ' ', s).strip()
toks = [w for w in TweetTokenizer().tokenize(s) if w not in STOP and len(w) > 1]
print('0 raw      :', ex[:150])
print('1-10 clean :', s[:150])
print('11-14 toks :', ' '.join(toks)[:150])
print('15 stemmed :', ' '.join(stemmer.stem(w) for w in toks)[:150])

#%% [B17] vocabulary effect - avg tokens before vs after cleaning
before = (df['designation'] + ' ' + df['description']).str.split().map(len).mean()
after = df['text'].str.split().map(len).mean()
print(f'avg tokens: raw {before:.1f} -> cleaned+stemmed {after:.1f}')

#%% [B22] preprocessing sanity asserts (text-level)
assert df['text'].notna().all(), 'NaN in cleaned text'
assert len(df) == 83502, 'unexpected row count after dedup'
assert (le.inverse_transform(df['y']) == df['prdtypecode'].values).all(), 'LabelEncoder round-trip failed'
#%% [B22b] missingness + length features, then remove exact duplicates (84,916 -> 83,502)
df['desc_missing']    = (df['description'].fillna('').astype(str).str.strip() == '').astype(int)
df['designation_len'] = df['designation'].fillna('').astype(str).str.len()
df['description_len'] = df['description'].fillna('').astype(str).str.len()
df = df.drop_duplicates(subset=['designation','description'], keep='first').reset_index(drop=True)
print('after dedup rows:', len(df))
df[['productid','imageid','designation','description','prdtypecode','y','text','text_nostem','desc_missing','designation_len','description_len']].to_csv(f'{DATA}/df_clean.csv', index=False)
print('text preprocessing OK ->', f'{DATA}/df_clean.csv')

# =============================================================================
# PHASE B - PREPROCESSING, IMAGE (B18-B21) - shown on one image; same steps for all
# (guarded: runs only if images.zip was downloaded/unzipped into data/images/image_train)
# =============================================================================
#%% [B18] load image by image_{imageid}_product_{productid}.jpg
r = df.iloc[0]
_imgpath = f"{DATA}/images/image_train/image_{r['imageid']}_product_{r['productid']}.jpg"
if os.path.exists(_imgpath):
    img = Image.open(_imgpath).convert('RGB')

    #%% [B19] corner-threshold crop of the white border
    a = np.array(img)
    corner = (a[0,0].mean() + a[0,-1].mean() + a[-1,0].mean() + a[-1,-1].mean()) / 4
    rows, cols, _ = np.where(a < corner*0.85)
    if rows.size:
        img = img.crop((cols.min(), rows.min(), cols.max(), rows.max()))

    #%% [B20] resize 224x224
    img = img.resize((224, 224))

    #%% [B21] data augmentation - random horizontal flip (training only)
    if np.random.RandomState(RS).rand() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    img.save(f'{FIG}/B19_B20_example_image.png')
    print('image preprocessing demo saved (B18-B21)')
else:
    print('B18-B21 image steps skipped - data/images/image_train not present (download images.zip for image phase)')
