import numpy as np
from utils import split_generator, Node
import argparse
class DecisionTree:
    """
    A decision tree classifier for use with real-valued attributes.
    """

    def __init__(self, max_depth=np.inf,mode = "gini"):
        """
        Decision tree constructor.

        :param max_depth: limit on the tree depth.
                          A depth 0 tree will have no splits.
        """
        self.max_depth = max_depth
        self._root = None  # Initialize the root node
        self.mode = "gini"

    def fit(self, X, y):
        """
        Construct the decision tree using the provided data and labels.

        :param X: Numpy array with shape (num_samples, num_features)
        :param y: Numpy integer array with length num_samples
        """
        self.classes = set(y)  # Store the unique classes in the dataset
        self._root = self._fit_tree(X, y, depth=0)

    def predict(self, X):
        """
        Predict labels for a data set by finding the appropriate leaf node for
        each input and using the majority label as the prediction.

        :param X:  Numpy array with shape (num_samples, num_features)
        :return: A length num_samples numpy array containing predicted labels.
        """
        if self._root is None:
            raise ValueError("Tree not fitted. Call 'fit' before 'predict'.")

        predictions = []
        for sample in X:
            node = self._root
            while not node.is_leaf():
                if sample[node.split.dim] <= node.split.pos:
                    node = node.left
                else:
                    node = node.right
            predictions.append(node.majority_class())

        return np.array(predictions)

    def get_depth(self):
        """
        :return: The depth of the decision tree.
        """
        return self._root._get_depth()

    def _fit_tree(self, X, y, depth):
        """
        Recursive method to fit the decision tree.

        :param X: Numpy array with shape (num_samples, num_features)
        :param y: Numpy integer array with length num_samples
        :param depth: Current depth of the tree
        :return: A Node object representing the subtree rooted at this node
        """
        if depth >= self.max_depth or len(set(y)) == 1:
            # If max depth is reached or all samples belong to the same class, create a leaf node
            return Node(labels = y)
        
        best_split = self._find_best_split(X, y)
        
        if best_split is None:
            # If no valid split is found, create a leaf node
            return Node(labels = y)
        
        X_left, y_left = best_split.X_left, best_split.y_left
        X_right, y_right = best_split.X_right, best_split.y_right

        # Recursively build left and right subtrees
        left_node = self._fit_tree(X_left, y_left, depth + 1)
        right_node = self._fit_tree(X_right, y_right, depth + 1)
        
        # Create a decision node for the best split
        return Node(best_split, left_node, right_node)

    def _find_best_split(self, X, y):
        """
        Find the best split for the current node based on Gini impurity.

        :param X: Numpy array with shape (num_samples, num_features)
        :param y: Numpy integer array with length num_samples
        :return: A Split object representing the best split or None if no valid split is found
        """
        if self.mode == "gini":
            best_split = None
            best_score = 1.0  # Initialize with a high value
            if len(list(set(y.tolist())))==1:
                return None
            import time
            for split in split_generator(X, y):
                score = (len(split.y_left)/len(y))*self._calculate_gini(split.y_left) + (len(split.y_right)/len(y))*self._calculate_gini(split.y_right)
                if score < best_score:
                    best_score = score
                    best_split = split
                if best_score==0:
                    return best_split
            

        elif self.mode=="ig":
            best_split = None
            best_score = 0  # Initialize with a high value
            if len(list(set(y.tolist())))==1:
                return None
            entropy_all = self._calculate_entropy(y)
            for split in split_generator(X, y):
                score = (len(split.y_left)/len(y))*self._calculate_entropy(split.y_left) + (len(split.y_right)/len(y))*self._calculate_entropy(split.y_right)
                score = entropy_all-score
                if score > best_score:
                    best_score = score
                    best_split = split
                if best_score==0:
                    return best_split
        else:
            raise Exception("Function does not exist.")
        return best_split

    def _calculate_gini(self, labels):
        """
        Calculate the Gini impurity of a set of labels.

        :param labels: Numpy integer array with length num_samples
        :return: Gini impurity
        """
        if len(labels) == 0:
            return 0.0
        
        gini = 1.0
        total_samples = len(labels)
        for class_label in self.classes:
            p_class = np.sum(labels == class_label) / total_samples
            gini -= np.square(p_class)
        
        return gini

    def _calculate_entropy(self, labels):
        """
        Calculate the Entropy of a set of labels.

        :param labels: Numpy integer array with length num_samples
        :return: Entropy
        """
        if len(labels) == 0:
            return np.inf
        
        entropy = 0
        total_samples = len(labels)
        for class_label in self.classes:
            p_class = np.sum(labels == class_label) / total_samples
            if p_class == 0:
                continue
            entropy += -p_class*np.log2(p_class)
        
        return entropy
class Node:
    """
    A node in the decision tree.

    Attributes:
        split (Split or None): A Split object representing the split at this node, or None for leaf nodes.
        left (Node or None): The left subtree (Node) of this node, or None for leaf nodes.
        right (Node or None): The right subtree (Node) of this node, or None for leaf nodes.
        labels (list): A list of labels for samples in the leaf node (only applicable for leaf nodes).
    """

    def __init__(self, split=None, left=None, right=None, labels=None):
        self.split = split
        self.left = left
        self.right = right
        self.labels = labels
        if self.is_leaf():
            self.labels = self.labels.tolist()
    def _get_depth(self):
        if self.is_leaf():
            return 0
        else:
            left_depth = self.left._get_depth() if self.left else 0
            right_depth = self.right._get_depth() if self.right else 0
            return max(left_depth, right_depth) + 1
    def is_leaf(self):
        """
        Check if this node is a leaf node.

        Returns:
            bool: True if the node is a leaf, False otherwise.
        """
        return self.split is None

    def majority_class(self):
        """
        Get the majority class in this node (only applicable for leaf nodes).

        Returns:
            int: The majority class label.
        """
        if self.is_leaf():
        
            return max(set(self.labels), key=self.labels.count)
        else:
            raise ValueError("majority_class can only be called on leaf nodes")




def tree_demo(mode):
    import draw_tree
    X = np.array([[0.88, 0.39],
                  [0.49, 0.52],
                  [0.68, 0.26],
                  [0.57, 0.51],
                  [0.61, 0.73]])
    y = np.array([1, 0, 0, 0, 1])
    tree = DecisionTree(mode=mode)
    tree.fit(X, y)
    draw_tree.draw_tree(X, y, tree)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a small demo to see if your code has any errors.')
    parser.add_argument('--demo', metavar='N', type=str,
                        help='Select a model to demo, "ig" for information gain, or "gini" for gini impurity.',
                        default="ig")

    args = parser.parse_args()
    tree_demo(args.demo)