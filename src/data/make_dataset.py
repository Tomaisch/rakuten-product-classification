"""
PHASE 0 - DOWNLOAD.
src/data/make_dataset.py - download the Rakuten challenge data.

What it does: downloads the text CSVs (and optionally the 2.5 GB image archive) into
data/, so the whole pipeline is reproducible from scratch on any machine. This is the
first phase of the pipeline; Phase A (descriptive) and Phase B (preprocessing) live in
src/features/build_features.py, Phase C (modeling) in benchmark_all_models.py.
Source: DataScientest / Rakuten challenge (challengedata.ens.fr).

Usage:  python src/data/make_dataset.py
"""
import os
import zipfile
from pathlib import Path

DATA = Path('data')
DATA.mkdir(exist_ok=True)

# Google-Drive file ids for the text files (mirror of the challenge data)
FILES = {
    'X_train_update.csv':  'REMOVED_GET_THIS_FILE_FROM_THE_CHALLENGE_PAGE',
    'Y_train_CVw08PX.csv': 'REMOVED_GET_THIS_FILE_FROM_THE_CHALLENGE_PAGE',
}
# The image archive (~2.5 GB) is downloaded from the challenge page; set IMAGES_ZIP_ID
# to a Drive id if you mirror it, otherwise place images.zip in data/ manually.
IMAGES_ZIP_ID = os.environ.get('RAKUTEN_IMAGES_ZIP_ID', '')


def download():
    import gdown
    for name, fid in FILES.items():
        out = DATA / name
        if out.exists():
            print(f'{name} already present - skip')
            continue
        print(f'downloading {name} ...')
        gdown.download(id=fid, output=str(out), quiet=False)

    zip_path = DATA / 'images.zip'
    if IMAGES_ZIP_ID and not zip_path.exists():
        print('downloading images.zip ...')
        import gdown
        gdown.download(id=IMAGES_ZIP_ID, output=str(zip_path), quiet=False)

    img_dir = DATA / 'images' / 'image_train'
    if zip_path.exists() and not img_dir.exists():
        print('unzipping images.zip ...')
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(DATA / 'images' if not (DATA / 'images').exists() else DATA)
    print('done. data/ now contains:', [p.name for p in DATA.iterdir()])


if __name__ == '__main__':
    download()
