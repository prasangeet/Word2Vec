from word2vec.cbow import CBOW
from word2vec.skip_gram import SkipGram

"""
this is the trainer class that creates and trains either CBOW or SkipGram
"""
class Trainer:

    def __init__(self, encoded_docs, vocab_size):
        """
        we store the encoded corpus and the vocab size
        the corpus is already tokenized and converted to indices
        """
        self.corpus     = encoded_docs
        self.vocab_size = vocab_size

    def create_model(self, model_type, dim=100, lr=0.025, negative=5):
        """
        we create either a CBOW or SkipGram model depending on what is passed in
        """
        if model_type == "cbow":
            return CBOW(vocab_size=self.vocab_size, embedding_dim=dim,
                        lr=lr, negative=negative)
        elif model_type == "skipgram":
            return SkipGram(vocab_size=self.vocab_size, embedding_dim=dim,
                            lr=lr, negative=negative)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def train(self, model_type, dim=100, window=5, negative=5, epochs=5, batch_size=512):
        """
        we create the model and run the training loop with the given hyperparameters
        """
        model = self.create_model(model_type=model_type, dim=dim, negative=negative)

        model.train(corpus=self.corpus, window=window, epochs=epochs, batch_size=batch_size)

        return model
