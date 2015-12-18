import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA, SparsePCA
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster


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

    def __init__(self, df, names=None):
        '''
            Set self.names and do preprocess data for dimension
            reduction.
        '''
        self.df = df
        self.columns = df.columns
        if names:
            self.names = names
        else:
            self.names = df.index

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

    def hcluster_cols(self, thresh):
        try:
            link = linkage(self.X.T, method='complete', metric='cosine')
            assignments = fcluster(link, thresh, 'distance')

        except:
            link = linkage(self.X.T, method='complete', metric='euclidean')
            assignments = fcluster(link, thresh, 'distance')

        col_ind = np.arange(len(self.columns))
        d = pd.DataFrame(zip(col_ind, assignments)).groupby(1)[0].aggregate(lambda x: tuple(x))
        df_new = pd.DataFrame(index=np.arange(len(self.names)))
        for i in d:
            cols = []
            for w in i:
                cols.append(w)
            if len(cols) > 1:
                df_new[str(self.columns[cols])] = np.mean(self.X[:, cols], axis=1)
            else:
                df_new[str(self.columns[cols[0]])] = self.X[:, cols[0]]

        self.df = df_new
        self.columns = df_new.columns.values

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
