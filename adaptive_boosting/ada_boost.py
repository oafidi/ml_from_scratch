import numpy as np
from decision_tree import DecisionTreeClassifier

class AdaBoostClassifier:
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.5) -> None:
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.trees: list[DecisionTreeClassifier] = []

    def _update_sample_weights(self,
                               X: np.ndarray,
                               y: np.ndarray,
                               y_pred: np.ndarray,
                               tree: DecisionTreeClassifier,
                               sample_weights: np.ndarray,) -> np.ndarray:

        signs = np.where(y_pred != y, 1, -1)
        weights = sample_weights * np.exp(tree.amount_of_say * signs)
        return weights / weights.sum()

    def _bootstrapping(self, X: np.ndarray, y: np.ndarray, sample_weights: np.ndarray) -> tuple:
        n_samples = X.shape[0]
        idxes = np.random.choice(n_samples, size=n_samples, replace=True, p=sample_weights)
        return X[idxes], y[idxes]

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X_bootstraped, y_bootstraped = X, y
        self.trees = []
        sample_weights = np.ones(X.shape[0]) / X.shape[0]

        for i in range(self.n_estimators):
            tree = DecisionTreeClassifier(max_depth=1)
            tree.fit(X_bootstraped, y_bootstraped)
            y_predicted = tree.predict(X)
            total_err = np.clip((sample_weights * (y_predicted != y)).sum(), 1e-9, 1-1e-9)
            tree.amount_of_say = self.learning_rate * np.log((1 - total_err) / total_err)
            self.trees.append(tree)

            sample_weights = self._update_sample_weights(X, y, y_predicted, tree, sample_weights)
            X_bootstraped, y_bootstraped = self._bootstrapping(X, y, sample_weights)

    def predict(self, X: np.ndarray) -> np.ndarray:
        votes = np.array([tree.predict(X) for tree in self.trees]).T  
        amounts = np.array([tree.amount_of_say for tree in self.trees])  

        classes = np.unique(votes)
        scores = np.zeros((X.shape[0], len(classes)))

        for j, cls in enumerate(classes):
            mask = (votes == cls)
            scores[:, j] = mask @ amounts
        return classes[np.argmax(scores, axis=1)]