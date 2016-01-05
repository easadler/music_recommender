from __future__ import division
import numpy as np
from scipy import sparse


class CollaborativeFilter(object):

    def __init__(self, similarity_metric=None, hood_size=None):
        self.similarity_metric = similarity_metric

    def fit(self, X):
        self.X = X
        self.pw = self.pairwise_jaccard(X)

    def pairwise_jaccard(self, X):
        """Computes the Jaccard distance between the rows of `X`.
        """
        X = X.astype(bool).astype(int).T

        intrsct = X.dot(X.T)
        row_sums = intrsct.diagonal()
        unions = row_sums[:, None] + row_sums - intrsct
        sim = intrsct / unions
        return sparse.csr_matrix(sim)

    def pred_all(self):
        pw = self.pw
        ratings_mat = self.X

        for u in xrange(ratings_mat.shape[0]):
            row = ratings_mat.getrow(u).toarray()[0]
            nz = row.nonzero()[0]
            z = np.nonzero(row == 0)[0]

            print 'user: ', u
            for i in z:
                denominator = pw[i, nz].sum()
                numerator = pw[i, nz].dot(ratings_mat[u, nz].T).toarray()[0]
                if denominator == 0:
                    print i, 0
                else:
                    print i, float(numerator / denominator)


if __name__ == '__main__':
    ratings_mat = sparse.csr_matrix(np.array([[4, 0, 0, 5, 1, 0, 0],
                                              [5, 5, 4, 0, 0, 0, 0],
                                              [0, 0, 0, 2, 4, 5, 0],
                                              [0, 3, 0, 0, 0, 0, 3]]))
    cf = CollaborativeFilter()
    cf.fit(ratings_mat)
