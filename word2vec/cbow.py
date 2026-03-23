import numpy as np
from word2vec.base_model import BaseWord2Vec
from tqdm import tqdm

"""
CBOW predicts the center word from the surrounding context words
we pre-build all training pairs and process them in batches
scatter-add replaces all inner loops so numpy stays fast
"""
class CBOW(BaseWord2Vec):

    def build_pairs(self, corpus, window):
        """
        we go through the whole corpus once and collect every
        center word and its context words into a list
        """
        centers  = []
        contexts = []
        for sentence in corpus:
            n = len(sentence)
            for i in range(n):
                start = max(0, i - window)
                end   = min(n, i + window + 1)
                ctx   = [sentence[j] for j in range(start, end) if j != i]
                if ctx:
                    centers.append(sentence[i])
                    contexts.append(ctx)
        return centers, contexts

    def train(self, corpus, window=5, epochs=5, batch_size=512):
        """
        we pre-build all pairs then process them in batches
        we also pre-compute the mean context vectors once before training starts
        """
        print("Building training pairs...")
        centers, contexts = self.build_pairs(corpus, window)
        n_pairs   = len(centers)
        centers_np = np.array(centers, dtype=np.int32)
        print(f"Total pairs: {n_pairs}")

        """
        pre-computing context means once saves recomputing them every epoch
        we refresh them after each epoch since V gets updated
        """
        print("Pre-computing context vectors...")
        ctx_means = np.array([self.V[ctx].mean(axis=0) for ctx in contexts], dtype=np.float32)

        for epoch in range(epochs):
            print(f"\nCBOW Epoch {epoch+1}/{epochs}")
            idx          = np.random.permutation(n_pairs)
            centers_shuf = centers_np[idx]
            ctx_shuf     = ctx_means[idx]
            ctx_list_shuf = [contexts[i] for i in idx]

            for bs in tqdm(range(0, n_pairs, batch_size), desc="Batches", leave=False):
                be       = min(bs + batch_size, n_pairs)
                c_batch  = centers_shuf[bs:be]
                ctx_batch = ctx_shuf[bs:be]                            # (B, dim)

                """
                positive update for the whole batch at once
                """
                u_c    = self.U[c_batch]                               # (B, dim)
                scores = np.sum(u_c * ctx_batch, axis=1)               # (B,)
                sigs   = self.sigmoid(scores)                          # (B,)
                errors = sigs - 1                                      # (B,)

                grad_u   = errors[:, None] * ctx_batch                 # (B, dim)
                grad_ctx = errors[:, None] * u_c                       # (B, dim)

                """
                negative updates for all pairs in the batch
                """
                neg_ids = np.array(
                    [self.negative_sampling(int(c)) for c in c_batch],
                    dtype=np.int32
                )                                                      # (B, neg)

                for k in range(self.negative):
                    nk    = neg_ids[:, k]
                    u_nk  = self.U[nk]                                 # (B, dim)
                    s_nk  = self.sigmoid(np.sum(u_nk * ctx_batch, axis=1))
                    grad_ctx += s_nk[:, None] * u_nk
                    np.add.at(self.U, nk, -(self.lr * s_nk[:, None] * ctx_batch))

                """
                scatter-add all gradients back with no Python loop
                """
                np.add.at(self.U, c_batch, -(self.lr * grad_u))

                for i, ctx in enumerate(ctx_list_shuf[bs:be]):
                    np.add.at(self.V, ctx, -(self.lr * grad_ctx[i] / len(ctx)))

            """
            refresh context means after each epoch since V has changed
            """
            ctx_means = np.array([self.V[ctx].mean(axis=0) for ctx in contexts], dtype=np.float32)
