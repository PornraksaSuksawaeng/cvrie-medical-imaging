# CVRIE

Epitech project done in a pair: supervised classification of brain MRI images, and unsupervised
clustering of 1,011 patient testimonies.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

The notebooks need two packages that are not in `requirements.txt` yet:

```bash
pip install jupyterlab opencv-python
```

TODO: add `jupyterlab` and `opencv-python` to `requirements.txt`.

## Project structure

```
data/raw/supervised/         brain MRI images, not in git
data/raw/unsupervised/       Student_Dataset.csv, the testimonies
figures/                     plots saved by the notebooks
notebooks/
  01_supervised.ipynb        image classification
  02_unsupervised.ipynb      testimony clustering
src/supervised/
  loader.py                  reads the images, labels come from the folder names
  preprocessing.py           grayscale, crop, resize, normalise
src/unsupervised/
  loader.py                  reads the malformed CSV
  preprocess.py              unicode cleaning and spaCy lemmas
  vectorize.py               TF-IDF and LSA
  cluster.py                 KMeans, Agglomerative, metrics
  plots.py                   t-SNE plots
scripts/cluster.py           command line clustering
requirements.txt
```

## Supervised part

Dataset: [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
from Kaggle, with 4 classes (glioma, meningioma, pituitary, notumor). It is not tracked in git,
so download it and unzip it into `data/raw/supervised/`, keeping its `Training` and `Testing`
folders.

Run the notebook:

```bash
jupyter lab notebooks/01_supervised.ipynb
```

TODO: there is no training script for this part, everything runs in the notebook.

## Unsupervised part

Run the notebook:

```bash
jupyter lab notebooks/02_unsupervised.ipynb
```

Run the clustering from the command line:

```bash
python -m scripts.cluster data/raw/unsupervised/Student_Dataset.csv clusters.csv
```

It writes one row per testimony (`id`, `color`, `cluster`) to `clusters.csv`, prints errors on
stderr and exits with code 84.

## Authors

- Pornraksa
- Leo Bellard
