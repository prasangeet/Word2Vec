from word2vec.skip_gram import SkipGram
from word2vec.cbow import CBOW

class Word2VecFactory:

    """
    The Model Factory that will create Word2Vec Model Objects
    This is the Factory Design Pattern, nothing too complicated.
    """

    @staticmethod
    def create(model_type, vocab_size, dim, lr=0.025, negative=5):

        if model_type == "cbow":
            return CBOW(vocab_size, dim, lr, negative)

        elif model_type == "skipgram":
            return SkipGram(vocab_size, dim, lr, negative)

        else:
            raise ValueError("Unknown model type")

