from gensim.models import Word2Vec
import logging

import os
from datetime import datetime

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
from src.preprocess import PreprocessPipeline
# from src.train_word2vec import Trainer

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
    f"results_gensim_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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

def train_word2vec(tokenized_docs):
    """
    Trains the original gensim Models
    We'll Compare them afterwards with the implemented models from scratch
    """

    logger.info("TRAINING GENSIM MODELS")

    configs = [
        (32, 3, 5),
        (32, 3, 10),
        (32, 5, 5),
        (32, 5, 10),
        (64, 3, 5),
        (64, 3, 10),
        (64, 5, 5),
        (64, 5, 10),
    ]

    models = {}

    for dim, window, neg in configs:

        logger.info(f"Training Gensim CBOW | dim = {dim}, window size = {window}, negatives = {neg}")

        cbow = Word2Vec(
            sentences=tokenized_docs,
            vector_size=dim,
            window=window,
            negative=neg,
            sg=0,
            min_count=1,
            workers=4,
            epochs=10
        )

        models[f"gensim_cbow_{dim}_{window}_{neg}"] = cbow

        logger.info(f"Training Gensim Skip-Gram | dim = {dim}, window = {window}, neg = {neg}")

        sg = Word2Vec(
            sentences=tokenized_docs,
            vector_size=dim,
            window=window,
            negative=neg,
            sg=1,
            min_count=1,
            workers=4,
            epochs=10
        )

        models[f"gensim_skipgram_{dim}_{window}_{neg}"] = sg

    return models

def semantic_analysis(models, words):
    """
    We'll find nearest neighbors using the Gensim models
    using built in similarity
    """

    logger.info("GENSIM NEAREST NEIGHBORS")

    for name, model in models.items():
        logger.info(f"\nModel: {name}")

        for word in words:
            logger.info(f"\nWord: {word}")

            if word not in model.wv:
                logger.info("Word not in vocabulary")
                continue

            neighbors = model.wv.most_similar(word, topn=5)

            for w, score in neighbors:
                logger.info(f"{w} {score:.4f}")

            logger.info("")

def analogy_gensim(models):
    """
    Analogy testing using Gensim.
    """

    logger.info("GENSIM ANALOGY")

    analogies = [
        ("undergraduate", "btech", "postgraduate"),
        ("btech", "student", "phd"),
        ("research", "faculty", "student"),
        ("course", "exam", "research"),
    ]

    for name, model in models.items():

        logger.info(f"\nModel: {name}")

        for a, b, c in analogies:

            logger.info(f"\n{a} : {b} :: {c} : ?")

            if not all(w in model.wv for w in [a, b, c]):
                logger.info("Word not in vocabulary")
                continue

            results = model.wv.most_similar(
                positive=[b, c],
                negative=[a],
                topn=5
            )

            for w, score in results:
                logger.info(f"{w} {score:.4f}")

        logger.info("")

def preprocess():
    """
    we run the preprocessing pipeline on the raw data
    it returns the vocab and the encoded documents
    """
    logger.info("DATASET STATISTICS")

    pipeline = PreprocessPipeline(DATA_DIR, CLEAN_CORPUS)

    documents, vocab, encoded_docs = pipeline.run()

    pipeline.save()

    total_docs, total_tokens, vocab_size = pipeline.stats()

    logger.info(f"Documents: {total_docs}")
    logger.info(f"Total Tokens: {total_tokens}")
    logger.info(f"Vocabulary Size: {vocab_size}")
    logger.info("")

    return vocab, encoded_docs

def main():
    """
    we run the gensim pipeline using the same preprocessing
    then train the gensim models and evaluate them
    """

    # we reuse the same preprocessing
    vocab, encoded_docs = preprocess()

    """
    gensim expects tokenized text not encoded integers
    so we reconstruct tokens using vocab
    """
    tokenized_docs = []
    for doc in encoded_docs:
        tokens = [vocab[idx] for idx in doc]
        tokenized_docs.append(tokens)

    # train gensim models
    gensim_models = train_word2vec(tokenized_docs)

    # run nearest neighbors
    words = ["research", "student", "phd", "exam"]
    semantic_analysis(gensim_models, words)

    # run analogies
    analogy_gensim(gensim_models)


if __name__ == "__main__":
    main()
