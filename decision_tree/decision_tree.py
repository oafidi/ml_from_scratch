import numpy as np
from tree_node import TreeNode

class DecisionTreeClassifier:
    def __init__(self, max_depth: int = 4, min_samples_leaf: int = 1, min_information_gain: float = 0.0):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_information_gain = min_information_gain
    
    def _class_probabilities(self, labels: list) -> list[float]:
        total_count = len(labels)
        counter = {}

        for label in labels:
            counter[label] = counter.get(label, 0) + 1

        return [label_count / total_count for label_count in counter.values()]

    def _entropy(self, class_probabilities: list[float]) -> float:
        return sum([-p * np.log2(p) for p in class_probabilities if p > 0])
    
    def _group_entropy(self, labels: list) -> float:
        return self._entropy(self._class_probabilities(labels))
    
    def _weighted_average_entropy(self, groups: list[list]) -> float:
        total_count = sum([len(group) for group in groups])

        return sum([self._group_entropy(group) * (len(group) / total_count) for group in groups])
    
    def _split(self, data: np.ndarray, feature_idx: int, feature_value: float) -> tuple:
        mask = data[:, feature_idx] < feature_value

        return data[mask], data[~mask]
    
    def _find_best_split(self, data: np.ndarray) -> dict:
        best_weighted_entropy = float("inf")
        best_feature_idx = -1
        best_threshold = None
        best_left_group, best_right_group = None, None

        for feature_idx in range(data.shape[1] - 1):
            thresholds = np.percentile(data[:, feature_idx], q=[25, 50, 75])
            
            for threshold in thresholds:
                left_group, right_group = self._split(data, feature_idx, threshold)
                
                weighted_entropy = self._weighted_average_entropy(
                    [left_group[:, -1], right_group[:, -1]]
                )
                
                if weighted_entropy < best_weighted_entropy:
                    best_weighted_entropy = weighted_entropy
                    best_feature_idx = feature_idx
                    best_threshold = threshold
                    best_left_group, best_right_group = left_group, right_group

        return {
            "weighted_entropy": best_weighted_entropy,
            "feature_idx": best_feature_idx,
            "threshold": best_threshold,
            "left_group": best_left_group,
            "right_group": best_right_group
        }
    
    def _build_tree(self, data: np.ndarray, current_depth: int = 0):

        if current_depth > self.max_depth:
            return None
        
        best_split_info = self._find_best_split(data)
        node_entropy = self._group_entropy(data[:, -1])
        information_gain = node_entropy - best_split_info["weighted_entropy"]
        
        node = TreeNode(best_split_info["feature_idx"],
                        best_split_info["threshold"],
                        information_gain
                    )

        if (information_gain < self.min_information_gain
                or len(best_split_info["left_group"]) < self.min_samples_leaf
                or len(best_split_info["right_group"]) < self.min_samples_leaf):
            return node
        
        node.right = self._build_tree(best_split_info["right_group"], current_depth + 1)
        node.left = self._build_tree(best_split_info["left_group"], current_depth + 1)

        return node


    

model = DecisionTreeClassifier()

print(model._entropy([0.5, 0.5]))