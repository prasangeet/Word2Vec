import numpy as np
from word2vec.base_model import BaseWord2Vec
from tqdm import tqdm

"""
SkipGram predicts the surrounding context words given a center word
we pre-build all pairs and use scatter-add to eliminate all inner Python loops
"""
class SkipGram(BaseWord2Vec):

    def build_pairs(self, corpus, window):
        """
        we flatten the whole corpus into a list of (center, context) int pairs
        """
        pairs = []
        for sentence in corpus:
            n = len(sentence)
            for i in range(n):
                start = max(0, i - window)
                end   = min(n, i + window + 1)
                for j in range(start, end):
                    if j != i:
                        pairs.append((sentence[i], sentence[j]))
        return np.array(pairs, dtype=np.int32)

    def train(self, corpus, window=5, epochs=5, batch_size=512):
        """
        we pre-build all pairs then process them in large batches
        all gradient updates use scatter-add so there are no Python loops
        inside the batch
        """
        print("Building training pairs...")
        pairs   = self.build_pairs(corpus, window)
        n_pairs = len(pairs)
        print(f"Total pairs: {n_pairs}")

        for epoch in range(epochs):
            print(f"\nSkipGram Epoch {epoch+1}/{epochs}")
            np.random.shuffle(pairs)

            for bs in tqdm(range(0, n_pairs, batch_size), desc="Batches", leave=False):
                be       = min(bs + batch_size, n_pairs)
                batch    = pairs[bs:be]
                centers  = batch[:, 0]                                 # (B,)
                contexts = batch[:, 1]                                 # (B,)

                """
                pull center and context vectors for the whole batch
                """
                v_c = self.V[centers]                                  # (B, dim)
                u_o = self.U[contexts]                                 # (B, dim)

                """
                positive sample gradients for the whole batch at once
                """
                scores  = np.sum(v_c * u_o, axis=1)                   # (B,)
                sigs    = self.sigmoid(scores)                         # (B,)
                errors  = sigs - 1                                     # (B,)

                grad_v  = errors[:, None] * u_o                        # (B, dim)
                grad_uo = errors[:, None] * v_c                        # (B, dim)

                """
                negative sample gradients for all B pairs
                """
                neg_ids = np.array(
                    [self.negative_sampling(int(c)) for c in centers],
                    dtype=np.int32
                )                                                      # (B, neg)

                for k in range(self.negative):
                    nk   = neg_ids[:, k]                               # (B,)
                    u_nk = self.U[nk]                                  # (B, dim)
                    s_nk = self.sigmoid(np.sum(v_c * u_nk, axis=1))    # (B,)
                    grad_v += s_nk[:, None] * u_nk
                    np.add.at(self.U, nk, -(self.lr * s_nk[:, None] * v_c))

                """
                scatter-add all positive gradients back with no Python loop
                """
                np.add.at(self.V, centers, -(self.lr * grad_v))
                np.add.at(self.U, contexts, -(self.lr * grad_uo))
