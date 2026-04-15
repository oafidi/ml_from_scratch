class TreeNode:
    def __init__(self, feature_idx: int = None, threshold: float = None, value: int = None):
        
        # for decision node
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.right = None
        self.left = None

        # for leaf node
        self.value = value