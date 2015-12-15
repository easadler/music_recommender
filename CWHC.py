import numpy as np
from collections import defaultdict
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import StandardScaler

import psycopg2 as pg2
import pandas as pd
import numpy as np


class CWHC(BaseEstimator, TransformerMixin):
    """Column-wise hierarchical clustering (CWHC)
    Column-wise hierarchical clustering of the
    data based on a threshold, which project the columns to a lower dimensional space.
    This implementation uses the scipy.cluster.hierarchy.linkage implementation hierachical clustering.
    It only works for dense arrays and is not scalable to large dimensional data.
    The time complexity of this implementation is ``O(n ** 3)`` assuming
    scipy's implementation uses the standard agglomerative clustering algorithm.

    Parameters
    ----------
    names : array (strings, ints), None
        Column names to be combined based on hierarchical clustering.
        if names is not set:
            names = np.arrange(n_features)
    thresh: float, None
        The threshold to apply when forming flat clusters in scipy's fcluster
    metric: numpy function
        The metric used to combine the column clusters.

    Notes
    -----
    Data should be scaled column-wise when using the fit, transform, and fit_transform methods.

    Examples
    --------
    >>> import CWHC
    >>> import pandas as pd
    >>> names = array(column names)
    >>> X = np.array(scaled data)
    >>> cwhc = CHWC(thresh = 0.3)
    >>> X_new = cwhc.fit_transform(X)
    >>> df = pd.DataFrame(X_new, columns = names)
    """

    def __init__(self, names=None, thresh=0.5, metric=np.mean):
        self.names = np.array(names)
        self.thresh = thresh
        self.metric = np.mean

    def fit(self, X, y=None):
        """Fit the model with X.
        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features.
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self._fit(X)
        return self

    def fit_transform(self, X, y=None):
        """Fit the model with X and apply the column-wise dimensionality reduction on X.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        X_new = self._fit(X)

        return X_new

    def transform(self, X):
        """Apply the clumn-wise dimensionality reduction on X.
        X columns are grouped into the clusters previous extracted
        from a training set.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples is the number of samples
            and n_features is the number of features.
        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        if self.X_new:
            return self.X_new

    def _fit(self, X):
        """Fit the model on X
        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.
        Returns
        -------
        X_new : array-like, shape (n_samples, n_features)
            The columns-wise clustering of the input data, copied and centered when
            requested.
        """
        col_ind = np.arange(X.shape[1])

        if self.names is None:
            self.names = col_ind

        # cluster and get assignments
        link = linkage(X.T, method='complete', metric='cosine')
        assignments = fcluster(link, self.thresh, 'distance')

        dic = defaultdict(list)

        # create dictionary of indexes based on assignment
        # reindex assinments to be zero-based
        for a, i in zip(assignments - 1, col_ind):
            dic[a].append(i)

        # initialize matrix with combined features and names array
        self.X_new = np.zeros((X.shape[0], len(dic)))
        self.new_names = []

        # create new columns from mean of cluster groups
        for k, v in dic.iteritems():
            self.new_names.append(str(self.names[v]))
            self.X_new[:, k] = self.metric(X[:, v], axis=1)

        return self.X_new

if __name__ == '__main__':
    # Connect to psql database
    conn = pg2.connect(dbname='lastfm', user='evansadler', host='/tmp')
    c = conn.cursor()
    query = 'SELECT * FROM sample;'
    df_t = pd.read_sql_query(query, conn)
    df_piv = df_t.groupby(['userid', 'artist'])['plays'].mean().reset_index().pivot(index='userid', columns='artist', values='plays')
    df_piv = df_piv[df_piv < 1000]

    summary = df_piv.dropna(thresh=70, axis=1)
    summary = summary.fillna(0)
    names = list(summary.columns)
    X = summary.values
    # ss = StandardScaler()
    # X = ss.fit_transform(X)
    cwhc = CWHC(thresh=0.8, names=names, metric=np.sum)
    X_new = cwhc.fit_transform(X)
    print cwhc.new_names
