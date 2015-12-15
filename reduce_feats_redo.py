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

    def __init__(self, model=None, scaler=None, cluster_model=False, thresh=0.5):
        '''
            Set self.names and do preprocess data for dimension
            reduction.
        '''
        self.model = model
        self.scaler = scaler
        self.cluster_model = cluster_model
        self.thresh = thresh

        if cluster_model:
            self.pipeline = Pipeline([('cluster_model', CWHC(thresh=0.5)), ('scaler', self.scaler), ('model', self.model)])
        else:
            self.pipeline = Pipeline([('scaler', self.scaler), ('model', self.model)])

    def fit(self, X, names=None):
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
        self.pipeline['cluster_model'].names = names
        self._fit(X)
        return self

    def _fit(self, X):
        self.X_new = self.pipeline.fit_transform(X)
        return self.X_new

    def fit_transform(self, X, names=None):
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
        self.fit(X)
        return self.X_new

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
            self.X_new = self.pipline.transform(X)
            return self.X_new

    def compenents(self, n_components):
        self.df_c = pd.DataFrame(self.pipline['model'].components_.T, index=self.names, columns=range(1, self.pipline['model'].n_components + 1))
        return self.df_c

    def best_cluster(self, n_cluster_list):
        best = (0,0, 0)
        for i in n_cluster_list:
            clusterer = KMeans(n_clusters=i)
            cluster_labels = clusterer.fit_predict(self.X)
            silhouette_avg = silhouette_score(self.X, cluster_labels)
            if abs(silhouette_avg) > best[1]:
                best = i, silhouette_avg, cluster_labels
            print "For n_clusters =", i, "The average silhouette_score is :", silhouette_avg
        self.best = best


    def plot_embedding(self, dimensions, figsize=(12, 12), name_lim=15, clusters):
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
