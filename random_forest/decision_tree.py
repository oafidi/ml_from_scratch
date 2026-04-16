import numpy as np
from tree_node import TreeNode

class DecisionTreeClassifier:
    def __init__(self, max_depth: int = 4, min_samples_leaf: int = 1, min_information_gain: float = 0.0, max_features: int = None):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_information_gain = min_information_gain
        self.max_features = max_features
        self.root = None
    
    def _class_probabilities(self, group: np.ndarray) -> list[float]:
        total_count = len(group)
        _, counts = np.unique(group, return_counts=True)

        return counts / total_count

    def _entropy(self, class_probabilities: np.ndarray) -> float:
        probabilities = class_probabilities[class_probabilities > 0]
        return np.sum(-probabilities * np.log2(probabilities))

    def _weighted_average_entropy(self, groups: list[np.ndarray]) -> float:
        total_count = sum([len(group) for group in groups])

        return sum([self._entropy(self._class_probabilities(group)) * (len(group) / total_count) for group in groups])
    
    def _information_gain(self, parent: np.ndarray, left_group: np.ndarray, right_group: np.ndarray) -> float:
        return self._entropy(self._class_probabilities(parent)) - self._weighted_average_entropy([left_group, right_group])

    def _split(self, data: np.ndarray, feature_idx: int, threshold: float) -> tuple:
        mask = data[:, feature_idx] < threshold

        return data[mask], data[~mask]
    
    def _find_best_split(self, data: np.ndarray, num_features: int) -> dict:

        best_split = {}
        best_split["information_gain"] = -float("inf")
        feature_idx_to_use = np.random.choice(list(range(num_features)),
                                              size=self.max_features if self.max_features else num_features,
                                              replace=False)

        for feature_idx in feature_idx_to_use:
            thresholds = np.unique(data[:, feature_idx])
            thresholds = (thresholds[: -1] + thresholds[1:]) / 2
            for threshold in thresholds:
                left_group, right_group = self._split(data, feature_idx, threshold)
                
                if len(left_group) > 0 and len(right_group) > 0:

                    information_gain = self._information_gain(data[:, -1], left_group[:, -1], right_group[:, -1])
                    if best_split["information_gain"] < information_gain:

                        best_split["feature_idx"] = feature_idx
                        best_split["threshold"] = threshold
                        best_split["left_group"] = left_group
                        best_split["right_group"] = right_group
                        best_split["information_gain"] = information_gain

        return best_split
    
    def _build_tree(self, data: np.ndarray, current_depth: int = 0) -> TreeNode:

        X, y = data[:, :-1], list(data[:, -1])
        num_samples, num_features = X.shape
        node = TreeNode()

        if num_samples < self.min_samples_leaf or current_depth > self.max_depth:
            node.value = max(y, key=y.count)
            return node

        best_split_info = self._find_best_split(data, num_features)

        if best_split_info["information_gain"] <= self.min_information_gain:
            node.value = max(y, key=y.count)
            return node

        node.feature_idx = best_split_info["feature_idx"]
        node.threshold = best_split_info["threshold"]
        node.right = self._build_tree(best_split_info["right_group"], current_depth + 1)
        node.left = self._build_tree(best_split_info["left_group"], current_depth + 1)

        return node

    def _predict(self, x, root) -> int:
        if root.value != None:
            return root.value
        feature_val = x[root.feature_idx]
        if feature_val < root.threshold:
            return self._predict(x, root.left)
        else:
            return self._predict(x, root.right)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        y = y.reshape(-1, 1)
        data = np.concatenate((X, y), axis=1)
        self.root = self._build_tree(data)

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.root != None, "Fit the model first"
        y_predicted = np.zeros(X.shape[0])

        for i, x in enumerate(X):
            y_predicted[i] = self._predict(x, self.root)
        return y_predicted.astype(int)
