import numpy as np
from decision_tree import DecisionTreeClassifier

class RandomForestClassifier:
    def __init__(self, n_estimators: int = 100, max_depth: int = 4, min_samples_leaf: int = 1, min_information_gain: float = 0.0, max_features: int | None = None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_information_gain = min_information_gain
        self.max_features = max_features
        self.trees = []

    def _create_bootstrap_samples(self, X: np.ndarray, y: np.ndarray) -> tuple:
        sample_size = X.shape[0]
        bootstrap_samples_X, bootstrap_samples_y = [], []

        for i in range(self.n_estimators):
            idxes = np.random.choice(sample_size, size=sample_size, replace=True)
            bootstrap_samples_X.append(X[idxes])
            bootstrap_samples_y.append(y[idxes])
        
        return bootstrap_samples_X, bootstrap_samples_y
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        bootstrap_samples_X, bootstrap_samples_y = self._create_bootstrap_samples(X, y)
        self.trees = []

        for i in range(self.n_estimators):
            tree = DecisionTreeClassifier(max_depth=self.max_depth,
                                          min_samples_leaf=self.min_samples_leaf,
                                          min_information_gain=self.min_information_gain,
                                          max_features=max(self.max_features, X.shape[1]) if self.max_features else int(np.sqrt(X.shape[1])))
            tree.fit(bootstrap_samples_X[i], bootstrap_samples_y[i])
            self.trees.append(tree)

    def predict(self, X: np.ndarray) -> np.ndarray:
        votes = np.array([tree.predict(X) for tree in self.trees]).T
        y_predicted = np.empty(X.shape[0])

        for i in range(X.shape[0]):
            labels, counts = np.unique(votes[i], return_counts=True)
            y_predicted[i] = labels[np.argmax(counts)]
        
        return y_predicted