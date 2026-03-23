import random
import numpy as np

"""
this is the base class for both CBOW and SkipGram
it holds the two embedding matrices V and U and the shared methods
"""
class BaseWord2Vec:

    def __init__(self, vocab_size, embedding_dim=100, lr=0.025, negative=5):
        """
        we initialize the two embedding matrices with small random values
        V is the input embeddings and U is the output embeddings
        """
        self.vocab_size    = vocab_size
        self.embedding_dim = embedding_dim
        self.lr            = lr
        self.negative      = negative

        scale  = 0.5 / embedding_dim
        self.V = np.random.uniform(-scale, scale, (vocab_size, embedding_dim))
        self.U = np.random.uniform(-scale, scale, (vocab_size, embedding_dim))

    def sigmoid(self, x):
        """
        standard sigmoid function to get probabilities between 0 and 1
        """
        return 1 / (1 + np.exp(-x))

    def negative_sampling(self, target):
        """
        we randomly pick words that are not the target word
        these are used as negative examples during training
        """
        negatives = []
        while len(negatives) < self.negative:
            sample = random.randint(0, self.vocab_size - 1)
            if sample != target:
                negatives.append(sample)
        return negatives

    def update(self, center, context):
        """
        this is the gradient update step used by skipgram
        we update the center and context embeddings using negative sampling
        """
        v_c = self.V[center]
        u_o = self.U[context]

        score = np.dot(u_o, v_c)
        sig   = self.sigmoid(score)

        grad_v  = (sig - 1) * u_o
        grad_uo = (sig - 1) * v_c
        self.U[context] -= self.lr * grad_uo

        """
        for each negative sample we compute the gradient and update
        we also accumulate the gradient for the center word
        """
        for neg in self.negative_sampling(context):
            u_k     = self.U[neg]
            sig_neg = self.sigmoid(np.dot(u_k, v_c))
            grad_v += sig_neg * u_k
            self.U[neg] -= self.lr * sig_neg * v_c

        self.V[center] -= self.lr * grad_v
