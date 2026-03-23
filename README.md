# NLU Assignment 2 - Problem 1
## Learning Word Embeddings from IIT Jodhpur Data

**Name:** Prasangeet Dongre  
**Roll Number:** B23CH1033

---

## What this does

This project trains Word2Vec models (CBOW and Skip-gram) from scratch on text collected from the IIT Jodhpur website. It then runs semantic analysis, analogy experiments, and visualizations on the learned embeddings.

---

## Project Structure

```
prob1/
│
├── datasets/
│   ├── pdf/                  raw downloaded PDFs and their extracted text
│   └── html/                 scraped HTML pages as text files
│
├── word2vec/
│   ├── base_model.py         base class with shared embedding matrices and update logic
│   ├── cbow.py               CBOW model
│   └── skip_gram.py          Skip-gram model
│
├── src/
│   ├── preprocess.py         preprocessing pipeline
│   └── train_word2vec.py     trainer wrapper
│
├── experiment_results/
│   └── images_DDMMYY_HHMMSS/   PCA and t-SNE plots saved here after each run
│
├── logs/                     training logs saved here with timestamps
├── models/                   trained model files
│
├── collect_data.py           scrapes and downloads all data from IITJ website
├── main.py                   runs everything: preprocess, train, analyze, visualize
├── wordcloud_gen.py          generates the word cloud from clean_corpus.txt
├── clean_corpus.txt          final cleaned corpus used for training
└── iitj_raw_corpus.txt       raw combined text before preprocessing
```

---

## Setup

**Python version:** 3.11

Install dependencies:

```bash
pip install -r requirements.txt
```

If you don't have a requirements file, install manually:

```bash
pip install numpy nltk scikit-learn matplotlib tqdm requests beautifulsoup4 pdfminer.six langdetect wordcloud
```

Download NLTK data (only needed once):

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

---

## How to run

### Step 1 — Collect the data

This scrapes the IIT Jodhpur website and downloads the PDFs. It saves everything to the `datasets/` folder and produces `iitj_raw_corpus.txt`.

```bash
python collect_data.py
```

### Step 2 — Run training and analysis

This runs the full pipeline: preprocessing, training all 16 models, semantic analysis, analogy experiments, and visualization.

```bash
python -m main
```

Logs are saved to `logs/results_YYYYMMDD_HHMMSS.log` and plots are saved to `experiment_results/images_DDMMYY_HHMMSS/`.

### Step 3 — Generate word cloud

```bash
python wordcloud_gen.py
```

Reads from `clean_corpus.txt` and saves `visualization_wordcloud.png`.

---

## What gets trained

The training grid runs every combination of:

| Parameter       | Values   |
|----------------|----------|
| Model type      | CBOW, Skip-gram |
| Embedding dim   | 32, 64   |
| Window size     | 3, 5     |
| Negatives       | 5, 10    |

This gives 16 models total. Each is trained for 5 epochs with batch size 512 and learning rate 0.025.

---

## Implementation notes

**No external ML libraries** are used for the models. CBOW and Skip-gram are implemented from scratch using only NumPy.

**Batched training** — all center-context pairs are pre-built before training starts and processed in batches of 512. Gradient updates use `np.add.at` for scatter-add so there are no slow Python loops inside the batch.

**Cosine similarity** is used for all nearest neighbour lookups and analogy experiments. Embeddings are L2-normalised before any dot product.

**Analogies** use the 3CosAdd formula: given words a, b, c the query vector is `(b - a + c) / norm` and the nearest word excluding a, b, c is returned.

---

## Output files

After running you will find:

- `clean_corpus.txt` — cleaned tokenized corpus, one document per line
- `logs/results_*.log` — full training log with semantic analysis and analogy results
- `experiment_results/images_*/pca_<model>.png` — PCA plots for each model
- `experiment_results/images_*/tsne_<model>.png` — t-SNE plots for each model
- `visualization_wordcloud.png` — word cloud from the corpus

---

## Results summary

Skip-gram with dim=32, window=5, neg=10 gave the best results. CBOW embeddings collapsed to near-identical cosine scores (~0.999) across unrelated words, making them not useful for semantic tasks on this corpus size. Skip-gram produced meaningful neighbours and correctly solved the BTech/MTech degree analogy.
