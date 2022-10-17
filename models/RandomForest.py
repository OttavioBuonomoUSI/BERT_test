from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

class RandomForest:
    def __init__(self, n_estimators=100, max_depth=2, random_state=0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = RandomForestClassifier(n_estimators=self.n_estimators,
                                            max_depth=self.max_depth,
                                            random_state=self.random_state)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def score(self, X, y):
        return self.model.score(X, y)

    def grid_search(self, X, y, param_grid):
        grid_search = GridSearchCV(self.model, param_grid, cv=5, n_jobs=-1)
        grid_search.fit(X, y)
        print("Best parameters: {}".format(grid_search.best_params_))
        print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))
        return grid_search

    def classification_report(self, X, y):
        y_pred = self.predict(X)
        print(classification_report(y, y_pred))

    def confusion_matrix(self, X, y):
        y_pred = self.predict(X)
        print(confusion_matrix(y, y_pred))
