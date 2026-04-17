class TreeNode:
    def __init__(self, feature_idx: int | None = None, threshold: float | None = None, value: int | None= None):
        
        # for decision node
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.right: TreeNode | None = None
        self.left: TreeNode | None = None

        # for leaf node
        self.value = value