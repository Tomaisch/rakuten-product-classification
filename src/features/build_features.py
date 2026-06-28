# =============================================================================
# PHASE A - DESCRIPTIVE ANALYSIS (16 steps) + PHASE B - PREPROCESSING (17 steps)
# Each step is its own independent #%% cell. Steps follow PREPROCESSING_STEPS.md.
# Reproducible: full 84,916 corpus, seed 42, no cache. Output: data/df_clean.csv
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
RS = 42; DATA = 'data'

#%% [load] merge X + Y
X = pd.read_csv(f'{DATA}/X_train_update.csv', index_col=0)
Y = pd.read_csv(f'{DATA}/Y_train_CVw08PX.csv', index_col=0)
df = X.join(Y)
df['designation'] = df['designation'].fillna(''); df['description'] = df['description'].fillna('')
assert len(df) == 84916, f'not full data: {len(df)}'
raw = df['designation'] + ' ' + df['description']

# PHASE A - DESCRIPTIVE ANALYSIS (16 steps)
# =============================================================================
#%% [A1] view the data
print(df.head(3)[['designation','description','productid','imageid','prdtypecode']])

#%% [A2] structure - shape, dtypes, cardinality
print(df.shape); print(df.dtypes); print(df.nunique())

#%% [A3] descriptive stats - char + word lengths
print('designation chars:', df['designation'].str.len().describe())
print('description chars:', df['description'].str.len().describe())
print('combined words   :', raw.str.split().map(len).describe())

#%% [A4] missing values per column
print((df[['designation','description']].apply(lambda c: (c.str.len()==0).sum())))

#%% [A5] missing-description rate BY CLASS + MNAR test (chi2 + Cramer's V)
no_desc = (df['description'].str.len()==0).astype(int)
ct = pd.crosstab(df['prdtypecode'], no_desc)
chi2, p, dof, _ = chi2_contingency(ct)
cramers_v = np.sqrt(chi2/(len(df)*(min(ct.shape)-1)))
print(f'MNAR chi2={chi2:.0f} p={p:.2e} CramersV={cramers_v:.3f}')

#%% [A6] class distribution + imbalance (figure)
vc = df['prdtypecode'].value_counts().sort_values(ascending=False)
plt.figure(figsize=(11,5)); plt.bar([str(c) for c in vc.index], vc.values)
plt.xticks(rotation=90, fontsize=7); plt.title(f'Class distribution (27, imbalance {vc.max()/vc.min():.1f}:1)')
plt.tight_layout(); plt.savefig('reports_A6_class_dist.png', dpi=120); plt.close()
print('imbalance ratio:', round(vc.max()/vc.min(),1))

#%% [A7] chi-square goodness-of-fit (is it uniform?)
print('GOF:', chisquare(vc.values))

#%% [A8] cumulative imbalance curve (figure)
cum = np.cumsum(vc.values)/vc.values.sum()
plt.figure(); plt.plot(range(1,28), cum, marker='o'); plt.xlabel('classes'); plt.ylabel('cumulative share')
plt.title('Cumulative imbalance'); plt.tight_layout(); plt.savefig('reports_A8_cumulative.png', dpi=120); plt.close()

#%% [A9] text-length distribution (figure)
plt.figure(figsize=(9,4)); plt.hist(raw.str.len().clip(upper=2000), bins=60)
plt.xlabel('combined length (chars, clipped 2000)'); plt.tight_layout(); plt.savefig('reports_A9_textlen.png', dpi=120); plt.close()

#%% [A10] Kruskal-Wallis - text length across classes
groups = [raw[df['prdtypecode']==c].str.len().values for c in vc.index]
print('Kruskal-Wallis:', kruskal(*groups))

#%% [A11] language detection + Wilson CIs (figure)
samp = raw.sample(2000, random_state=RS)
def _safe(t):
    try: return detect(t[:200])
    except: return 'unknown'
langs = samp.map(_safe).value_counts()
print((langs/len(samp)*100).head(6))
top = langs.head(6)
ci = [proportion_confint(n, len(samp), method='wilson') for n in top.values]
print('Wilson CIs:', [(round(a,3),round(b,3)) for a,b in ci])

#%% [A12] top tokens overall
from collections import Counter
print(Counter(' '.join(raw.sample(5000, random_state=RS)).lower().split()).most_common(20))

#%% [A13] duplicates
print('dup combined-text rows:', int(raw.duplicated().sum()))

#%% [A14] image inspection - size / white border (sample)
from PIL import Image
r0 = df.iloc[0]; p0 = f"{DATA}/images/image_train/image_{r0['imageid']}_product_{r0['productid']}.jpg"
if os.path.exists(p0):
    im0 = Image.open(p0); print('example image size:', im0.size)

#%% [A15] target encoding map (codes -> 0..26)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder(); df['y'] = le.fit_transform(df['prdtypecode'])
print('classes:', len(le.classes_))

#%% [A16] decisions taken from the EDA
print('Decisions: merge text; keep ALL languages; rank by macro-F1; class-weight/SMOTE for imbalance.')

# =============================================================================

# PHASE B - PRE-PROCESSING (TEXT, 15 steps; KEEPS ALL ROWS)
# =============================================================================
#%% [B1] merge designation + description
text = df['designation'] + ' ' + df['description']

#%% [B2] lowercase
text = text.str.lower()

#%% [B3] strip URLs
text = text.map(lambda t: re.sub(r'http\S+|www\S+', ' ', t))

#%% [B4] strip emails
text = text.map(lambda t: re.sub(r'\S+@\S+', ' ', t))

#%% [B5] strip HTML tags - BeautifulSoup
text = text.map(lambda t: BeautifulSoup(t, 'html.parser').get_text())

#%% [B6] strip HTML entities (&eacute;)
text = text.map(lambda t: re.sub(r'&[a-z]+;', ' ', t))

#%% [B7] strip sizes "8 x 5"
text = text.map(lambda t: re.sub(r'\d+\s*x\s*\d+', ' ', t))

#%% [B8] strip bare numbers
text = text.map(lambda t: re.sub(r'\d+', ' ', t))

#%% [B9] strip special chars, keep French accents
text = text.map(lambda t: re.sub(r'[^a-zàâäéèêëïîôöùûüçœ ]', ' ', t))

#%% [B10] collapse whitespace
text = text.map(lambda t: re.sub(r'\s+', ' ', t).strip())

#%% [B11] tokenize - TweetTokenizer
tok = TweetTokenizer()
tokens = text.map(tok.tokenize)

#%% [B12] keep ALL languages - no row dropping
#

#%% [B13] remove stopwords FR + EN + DE
STOP = set(stopwords.words('french')) | set(stopwords.words('english')) | set(stopwords.words('german'))
tokens = tokens.map(lambda ws: [w for w in ws if w not in STOP])

#%% [B14] drop length-1 tokens
tokens = tokens.map(lambda ws: [w for w in ws if len(w) > 1])

#%% [B15] Snowball-French stemming (classical branch; OFF for transformers)
stemmer = SnowballStemmer('french')
df['text']        = tokens.map(lambda ws: ' '.join(stemmer.stem(w) for w in ws))   # classical / TF-IDF
df['text_nostem'] = tokens.map(lambda ws: ' '.join(ws))                            # transformers
df[['productid','imageid','y','text','text_nostem']].to_csv(f'{DATA}/df_clean.csv', index=False)
print('text preprocessing done ->', f'{DATA}/df_clean.csv')

# =============================================================================

# PHASE B - PRE-PROCESSING (IMAGE, steps 16-17)
# =============================================================================
#%% [B16] corner-threshold crop of the white border
def crop_white(path):
    im = Image.open(path).convert('RGB'); a = np.array(im)
    corner = (a[0,0].mean()+a[0,-1].mean()+a[-1,0].mean()+a[-1,-1].mean())/4
    rows, cols, _ = np.where(a < corner*0.85)
    if rows.size: im = im.crop((cols.min(), rows.min(), cols.max(), rows.max()))
    return im

#%% [B17] resize 224x224
def load_image(imageid, productid):
    im = crop_white(f"{DATA}/images/image_train/image_{imageid}_product_{productid}.jpg")
    return im.resize((224,224))

# =============================================================================
