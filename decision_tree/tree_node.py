class TreeNode:
    def __init__(self, feature_idx: int, threshold: float, class_probabilities: list[float], information_gain: float):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.information_gain = information_gain
        self.right = None
        self.left = None