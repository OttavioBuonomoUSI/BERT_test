import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from sklearn.neural_network import MLPClassifier

from train import get_df


class Neur_classifier(BaseEstimator):

    def __init__(self) -> None:
        super().__init__()
        self.__is_fitted__ = False

    def fit(self, X, y):
        super(Neur_classifier, self).fit()
        pass

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)

    def __sklearn_is_fitted__(self):
        return self.__is_fitted__


def train_nn(x, y):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(5, 2), random_state=1)

    return clf


if __name__ == '__main__':
    df = get_df()
    param_grid = {'C': [0.1, 1, 10, 100],
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                  'gamma': ['scale', 'auto'],
                  'kernel': ['linear']}
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(5, 2), random_state=1)
    grid = GridSearchCV(clf, param_grid, refit=True, verbose=3, n_jobs=-1)
    pass
