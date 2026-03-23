import os
import logging
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.preprocess import PreprocessPipeline
from src.train_word2vec import Trainer

"""
all the constants we need to run the code
"""
DATA_DIR = "datasets"
CLEAN_CORPUS = "clean_corpus.txt"
MODEL_DIR = "models"
LOG_DIR = "logs"

os.makedirs(LOG_DIR, exist_ok=True)

log_file = os.path.join(
    LOG_DIR,
    f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)

"""
we save the logs to a file and also print them to the console
we use datetime in the filename so each run gets its own log file
"""
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger()


def preprocess():
    """
    we run the preprocessing pipeline on the raw data
    it returns the vocab and the encoded documents
    """
    logger.info("===== DATASET STATISTICS =====")

    pipeline = PreprocessPipeline(DATA_DIR, CLEAN_CORPUS)

    documents, vocab, encoded_docs = pipeline.run()

    pipeline.save()

    total_docs, total_tokens, vocab_size = pipeline.stats()

    logger.info(f"Documents: {total_docs}")
    logger.info(f"Total Tokens: {total_tokens}")
    logger.info(f"Vocabulary Size: {vocab_size}")
    logger.info("")

    return vocab, encoded_docs


def train(vocab, encoded_docs):
    """
    we train both CBOW and SkipGram across different combinations
    of embedding dimensions, window sizes and number of negatives
    to find what works best
    """
    trainer = Trainer(
        encoded_docs=encoded_docs,
        vocab_size=len(vocab)
    )

    dimensions = [32, 64]
    windows = [3, 5]
    negatives = [5, 10]

    os.makedirs(MODEL_DIR, exist_ok=True)

    trained_models = {}

    for dim in dimensions:
        for window in windows:
            for neg in negatives:

                logger.info(f"Training CBOW | dim={dim} window={window} neg={neg}")

                cbow = trainer.train(
                    model_type="cbow",
                    dim=dim,
                    window=window,
                    negative=neg
                )

                trained_models[f"cbow_{dim}_{window}_{neg}"] = cbow

                logger.info(f"Training SkipGram | dim={dim} window={window} neg={neg}")

                sg = trainer.train(
                    model_type="skipgram",
                    dim=dim,
                    window=window,
                    negative=neg
                )

                trained_models[f"skipgram_{dim}_{window}_{neg}"] = sg

    return trained_models


def normalize_embeddings(embeddings):
    """
    we l2 normalize each row so we can use dot product as cosine similarity
    we add a small epsilon to avoid dividing by zero
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / (norms + 1e-10)


def semantic_analysis(models, vocab):
    """
    for each model we find the 5 most similar words to a given word
    we use cosine similarity by normalizing the embeddings first
    """
    word2idx = {w: i for i, w in enumerate(vocab)}
    words = ["research", "student", "phd", "exam"]

    for name, model in models.items():
        logger.info(f"\nModel: {name}")

        embeddings = normalize_embeddings(model.V)

        for word in words:
            logger.info(f"\nWord: {word}")
            if word not in word2idx:
                logger.info("Word not in vocabulary")
                continue

            idx = word2idx[word]
            vec = embeddings[idx]

            """
            since both the embeddings and vec are normalized
            the dot product gives us the cosine similarity directly
            we exclude the word itself by setting its score to -inf
            """
            similarities = embeddings @ vec
            similarities[idx] = -np.inf

            nearest = np.argsort(-similarities)[:5]
            for n in nearest:
                logger.info(f"{vocab[n]} {similarities[n]:.4f}")

        logger.info("")


def analogy_stage(models, vocab):
    """
    we test word analogies using the formula b - a + c
    for example btech is to student as phd is to what
    we use cosine similarity to find the closest word to the result vector
    """
    logger.info("===== ANALOGY EXPERIMENTS =====")

    word2idx = {w: i for i, w in enumerate(vocab)}

    analogies = [
        ("undergraduate", "btech", "postgraduate"),
        ("btech", "student", "phd"),
        ("research", "faculty", "student"),
        ("course", "exam", "research"),
    ]

    for name, model in models.items():
        logger.info(f"\nModel: {name}")

        normalized_embeddings = normalize_embeddings(model.V)

        for a, b, c in analogies:
            logger.info(f"\n{a} : {b} :: {c} : ?")

            if not all(w in word2idx for w in [a, b, c]):
                logger.info("Word not in vocabulary")
                continue

            """
            we compute the analogy vector using normalized embeddings
            then we normalize the result before doing the similarity search
            """
            vec = (
                normalized_embeddings[word2idx[b]]
                - normalized_embeddings[word2idx[a]]
                + normalized_embeddings[word2idx[c]]
            )

            vec_norm = vec / (np.linalg.norm(vec) + 1e-10)

            similarities = normalized_embeddings @ vec_norm

            """
            we pick the top 5 results but skip the three words
            that were used to build the analogy vector
            """
            nearest = []
            for idx in np.argsort(-similarities):
                if vocab[idx] not in {a, b, c}:
                    nearest.append(idx)
                if len(nearest) == 5:
                    break

            for n in nearest:
                logger.info(f"{vocab[n]} {similarities[n]:.4f}")

        logger.info("")


def visualization(models, vocab):
    """
    we plot the word embeddings in 2d using PCA and tSNE
    for a small set of words to see how they cluster together
    we save one plot per model
    """
    logger.info("===== TASK 4: VISUALIZATION =====")

    curr_time = datetime.now().strftime("_%d%m%y_%H%M%S")
    IMAGE_DIR = f"experiment_results/images{curr_time}"
    os.makedirs(IMAGE_DIR, exist_ok=True)

    selected_words = [
        "research","student","phd","faculty",
        "engineering","science","technology",
        "course","exam","program"
    ]

    word2idx = {w: i for i, w in enumerate(vocab)}

    for name, model in models.items():

        logger.info(f"Visualizing model: {name}")

        embeddings = model.V

        words = []
        vectors = []

        for w in selected_words:
            if w in word2idx:
                words.append(w)
                vectors.append(embeddings[word2idx[w]])

        if len(vectors) < 2:
            continue

        vectors = np.array(vectors)

        """
        PCA reduces the embeddings to 2 dimensions
        we plot and label each word then save the figure
        """
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(vectors)

        plt.figure(figsize=(8,6))

        for i, word in enumerate(words):
            x, y = pca_result[i]
            plt.scatter(x, y)
            plt.text(x+0.01, y+0.01, word)

        plt.title(f"PCA - {name}")
        plt.savefig(f"{IMAGE_DIR}/pca_{name}.png")
        plt.close()

        """
        tSNE also reduces to 2 dimensions but captures non linear structure better
        we use perplexity 5 since we only have a small number of words
        """
        tsne = TSNE(n_components=2, random_state=42, perplexity=5)
        tsne_result = tsne.fit_transform(vectors)

        plt.figure(figsize=(8,6))

        for i, word in enumerate(words):
            x, y = tsne_result[i]
            plt.scatter(x, y)
            plt.text(x+0.01, y+0.01, word)

        plt.title(f"t-SNE - {name}")
        plt.savefig(f"{IMAGE_DIR}/tsne_{name}.png")
        plt.close()


def main():
    """
    we run all the steps in order
    preprocess the data, train the models, run the analysis and visualize
    """
    vocab, encoded_docs = preprocess()

    models = train(vocab, encoded_docs)

    semantic_analysis(models, vocab)

    analogy_stage(models, vocab)

    visualization(models, vocab)


if __name__ == "__main__":
    main()
