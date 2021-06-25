from __future__ import absolute_import
from numpy.linalg import cholesky
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array
import numpy as np

class BaseMetricLearner(BaseEstimator, TransformerMixin):
  def __init__(self):
    raise NotImplementedError('BaseMetricLearner should not be instantiated')

  def metric(self):
    """Computes the Mahalanobis matrix from the transformation matrix.

    .. math:: M = L^{\\top} L

    Returns
    -------
    M : (d x d) matrix
    """
    L = self.transformer()
    return L.T.dot(L)

  def transformer(self):
    """Computes the transformation matrix from the Mahalanobis matrix.

    L = cholesky(M).T

    Returns
    -------
    L : upper triangular (d x d) matrix
    """
    return cholesky(self.metric()).T

  def transform(self, X=None):
    """Applies the metric transformation.

    Parameters
    ----------
    X : (n x d) matrix, optional
        Data to transform. If not supplied, the training data will be used.

    Returns
    -------
    transformed : (n x d) matrix
        Input data transformed to the metric space by :math:`XL^{\\top}`
    """
    if X is None:
      X = self.X_
    else:
      X = check_array(X, accept_sparse=True)
    L = self.transformer()
    return X.dot(L.T)



class Euclidean(BaseMetricLearner):
    def __init__(self):
        self.M_ = None

    def metric(self):
        return self.M_

    def fit(self, X):
        self.M_ = np.eye(X.shape[1])
        self.X_ = X

    def transform(self, X=None):
        if X is None:
            return self.X_
        return X

    def get_metric(self):
        pass
    def score_pairs(self):
        pass


a = Euclidean()