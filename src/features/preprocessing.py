"""
PHASE B - PREPROCESSING (shared, common module).
src/features/preprocessing.py - the ONE preprocessing every team member imports, so all
models (classical, transformer, image) train on identical cleaned inputs.

This is the agreed 39-step union (see README section 4 / PREPROCESSING_STEPS.md):
  TEXT  (steps B1-B15): merge designation+description -> lowercase -> strip URLs/emails/
        HTML tags/HTML entities/dimensions "8 x 5"/bare numbers/special chars (keep FR
        accents) -> collapse whitespace -> TweetTokenizer -> drop FR+EN+DE stopwords +
        len>1 -> optional Snowball French stemming. KEEPS ALL ROWS / ALL LANGUAGES (B12).
  IMAGE (steps B16-B19): load by image_{imageid}_product_{productid}.jpg -> autocrop white
        borders -> resize 224x224. (Augmentation/flips are applied in the training loader.)

Usage:
    from src.features.preprocessing import clean_text, tokenize_clean, clean_image
    df['text'] = (df['designation'] + ' ' + df['description']).map(clean_text).map(
                  lambda s: tokenize_clean(s, stem=True))   # stem=True  -> TF-IDF / classical
                                                            # stem=False -> transformers (CamemBERT...)
    img = clean_image(IMG_DIR / f"image_{imageid}_product_{productid}.jpg")
"""
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem.snowball import SnowballStemmer

# FR + EN + DE stopwords (B13) - keeps German rows usable instead of dropping non-FR/EN text
STOP_WORDS = (set(stopwords.words('french'))
              | set(stopwords.words('english'))
              | set(stopwords.words('german')))
_TOKENIZER = TweetTokenizer()
_STEMMER = SnowballStemmer('french')


def clean_text(t):
    """Text steps B2-B10 (B1 merge done by caller; B12 keep-all-languages = no row dropping)."""
    t = (t or '').lower()                              # B2  lowercase
    t = re.sub(r'http\S+|www\S+', ' ', t)              # B3  strip URLs
    t = re.sub(r'\S+@\S+', ' ', t)                     # B4  strip emails
    t = re.sub(r'<.*?>', ' ', t)                       # B5  strip HTML tags
    t = re.sub(r'&[a-z]+;', ' ', t)                    # B6  strip HTML entities (&eacute;)
    t = re.sub(r'\d+\s*x\s*\d+', ' ', t)               # B7  strip dimensions "8 x 5"
    t = re.sub(r'\d+', ' ', t)                         # B8  strip bare numbers
    t = re.sub(r'[^a-zàâäéèêëïîôöùûüçœ ]', ' ', t)     # B9  strip special chars (keep FR accents)
    return re.sub(r'\s+', ' ', t).strip()              # B10 collapse whitespace


def tokenize_clean(t, stem=True):
    """Text steps B11-B15. stem=True for TF-IDF/classical; stem=False for transformers."""
    toks = _TOKENIZER.tokenize(t)                                   # B11 TweetTokenizer
    toks = [w for w in toks if w not in STOP_WORDS and len(w) > 1]  # B13 stopwords FR+EN+DE, B14 len>1
    if stem:
        toks = [_STEMMER.stem(w) for w in toks]                     # B15 Snowball French stemming
    return ' '.join(toks)


def preprocess_text(designation, description, stem=True):
    """Convenience: B1 merge + full text pipeline in one call."""
    return tokenize_clean(clean_text(f"{designation or ''} {description or ''}"), stem=stem)


def clean_image(path, size=224):
    """Image steps B16-B18: load -> autocrop white borders -> resize size x size (PIL.Image)."""
    from PIL import Image
    image = Image.open(path).convert('RGB')            # B16 load
    a = np.array(image)
    corner = (a[0, 0].mean() + a[0, -1].mean() + a[-1, 0].mean() + a[-1, -1].mean()) / 4
    rows, cols, _ = np.where(a < corner * 0.85)        # B17 autocrop white borders
    if rows.size == 0:
        return image.resize((size, size))              # B18 resize
    t, b, l, r = rows.min(), rows.max(), cols.min(), cols.max()
    if t != b and l != r:
        image = image.crop((l, t, r, b))
    return image.resize((size, size))                  # B18 resize 224x224
