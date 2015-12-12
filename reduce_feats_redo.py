import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

from sklearn.decomposition import PCA, SparsePCA
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.cluster.hierarchy import linkage, fcluster


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

    Notes
    -----
    For n_components='mle', this class uses the method of `Thomas P. Minka:
    Automatic Choice of Dimensionality for PCA. NIPS 2000: 598-604`
    Implements the probabilistic PCA model from:
    M. Tipping and C. Bishop, Probabilistic Principal Component Analysis,
    Journal of the Royal Statistical Society, Series B, 61, Part 3, pp. 611-622
    via the score and score_samples methods.
    See http://www.miketipping.com/papers/met-mppca.pdf
    Due to implementation subtleties of the Singular Value Decomposition (SVD),
    which is used in this implementation, running fit twice on the same matrix
    can lead to principal components with signs flipped (change in direction).
    For this reason, it is important to always use the same estimator object to
    transform data in a consistent fashion.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.decomposition import PCA
    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> pca = PCA(n_components=2)
    >>> pca.fit(X)
    PCA(copy=True, n_components=2, whiten=False)
    >>> print(pca.explained_variance_ratio_) # doctest: +ELLIPSIS
    [ 0.99244...  0.00755...]
    See also
    --------
    RandomizedPCA
    KernelPCA
    SparsePCA
    TruncatedSVD
    """

    def __init__(self, names=None, thresh=0.5):
        self.names = names
        self.threshold = thresh

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

        if not self.names:
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
            self.X_new[:, k] = np.mean(X[:, v], axis=1)

        return self.X_new


class ReduceFeatures(object):
    """Make 2d or 3d plots of principle components from scikit learn's PCA or
    SparsePCA models. Option to use scikit learn's KMeans model to color observations.

    Parameters
    ----------
    df : dataframe
    min_samples : int, optional
        The number of samples (or total weight) in a neighborhood for a point
        to be considered as a core point. This includes the point itself.
    metric : string, or callable
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string or callable, it must be one of
        the options allowed by metrics.pairwise.calculate_distance for its
        metric parameter.
        If metric is "precomputed", X is assumed to be a distance matrix and
        must be square. X may be a sparse matrix, in which case only "nonzero"
        elements may be considered neighbors for DBSCAN."""

    def __init__(self, model=None, scaler=None, cluster_model=False, col_cluster_thresh=None):
        '''
            Set self.names and do preprocess data for dimension
            reduction.
        '''
        self.model = model
        self.scaler = scaler
        self.cluster_model = cluster_model
        self.col_cluster_thresh = 0.1

        if col_cluster_thresh:
            self.pipeline = Pipeline([('scaler', self.scaler), ('col_clust_model', self.CWHC), ('model', self.model)])
        else:
            self.pipeline = Pipeline([('scaler', self.scaler), ('model', self.model)])

    def fit(self, X):
        self.X = self.pipeline(X)

    def fit_transform(self, X):
        return self.model.fit_transform(X)

    def preprocessing(self, non_na_thresh=None, na_fill_char=None):
        if non_na_thresh is None:
            non_na_thresh = 0
        mask = (self.df.dtypes == np.float64) | (self.df.dtypes == np.int_)
        df_sub = self.df.ix[:, mask]
        df_sub = df_sub.dropna(axis=1, thresh=int(self.df.shape[1] * non_na_thresh))

        if na_fill_char:
            df_sub = df_sub.fillna(na_fill_char)

        imp = preprocessing.Imputer(axis=0)
        X = imp.fit_transform(df_sub)
        X_centered = preprocessing.scale(X)

        self.X = X_centered
        self.columns = df_sub.columns.values

    def fit_pca(self, n_components):
        pca = PCA(n_components=n_components)
        self.X = pca.fit_transform(self.X)
        self.df_c = pd.DataFrame(pca.components_.T, index=self.crimes, columns=range(1, n_components + 1))

        print pca.explained_variance_ratio_
        return self.df_c

    def sparse_pca(self, n_components, alpha):
        pca = SparsePCA(n_components=3, alpha=alpha, n_jobs=-1)
        self.X = pca.fit_transform(self.X)
        self.df_c = pd.DataFrame(pca.components_.T, index=self.crimes, columns=range(1, n_components + 1))

        return self.df_c

    def best_cluster(self):
        best = (0, 0, 0)
        for i in [3, 4, 5, 6, 7, 8, 9, 10]:
            clusterer = KMeans(n_clusters=i)
            cluster_labels = clusterer.fit_predict(self.X)
            silhouette_avg = silhouette_score(self.X, cluster_labels)
            if abs(silhouette_avg) > best[1]:
                best = i, silhouette_avg, cluster_labels
            print "For n_clusters =", i, "The average silhouette_score is :", silhouette_avg
        self.best = best

    def plot_embedding(self, dimensions, figsize=(12, 12), name_lim=15):
        y = self.best[2]
        X = self.X

        if dimensions == 3:
            fig = plt.figure(figsize=figsize, dpi=250)
            ax = fig.add_subplot(111, projection='3d')

            for i in range(X.shape[0]):
                ax.text(X[i, 0], X[i, 1], X[i, 2], str(self.names[i][0:name_lim]), color=plt.cm.Set1(y[i] / 10.), fontsizes=8)
            ax.set_xlim3d(X[:, 0].min(), X[:, 0].max())
            ax.set_ylim3d(X[:, 1].min(), X[:, 1].max())
            ax.set_zlim3d(X[:, 2].min(), X[:, 2].max())
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
        elif dimensions == 2:
            plt.figure(figsize=(12, 12), dpi=250)
            ax = plt.subplot(111)

            for i in range(X.shape[0]):
                ax.text(X[i, 0], X[i, 1], str(self.names[i][0:name_lim]), color=plt.cm.Set1(y[i] / 10.), fontsize=8)
            ax.set_xlim(X[:, 0].min(), X[:, 0].max())
            ax.set_ylim(X[:, 1].min(), X[:, 1].max())
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
        plt.show()
