from collections import namedtuple
import numpy as np
# Named tuple is a quick way to create a simple wrapper class...
Split_ = namedtuple('Split',
                    ['dim', 'pos', 'X_left', 'y_left', 'X_right', 'y_right'])

class Split(Split_):
    """
    Represents a possible split point during the decision tree creation process.

    Attributes:

        dim (int): the dimension along which to split
        pos (float): the position of the split
        X_left (ndarray): all X entries that are <= to the split position
        y_left (ndarray): labels corresponding to X_left
        X_right (ndarray):  all X entries that are > the split position
        y_right (ndarray): labels corresponding to X_right
    """
    pass

def split_generator(X, y):
    """
    Utility method for generating all possible splits of a data set
    for the decision tree construction algorithm.

    :param X: Numpy array with shape (num_samples, num_features)
    :param y: Numpy integer array with length num_samples
    :return: A generator for Split objects that will yield all
            possible splits of the data
    """

    # Loop over all of the dimensions.
    for dim in range(X.shape[1]):
        # Get the indices in sorted order so we can sort both  data and labels
        ind = np.argsort(X[:, dim])

        # Copy the data and the labels in sorted order
        X_sort = X[ind, :]
        y_sort = y[ind]

        # Loop through the midpoints between each point in the current dimension
        for index in range(1, X_sort.shape[0]):

            # don't try to split between equal points.
            if X_sort[index - 1, dim] != X_sort[index, dim]:
                pos = (X_sort[index - 1, dim] + X_sort[index, dim]) / 2.0

                # Yield a possible split.  Note that the slicing here does
                # not make a copy, so this should be relatively fast.
                yield Split(dim, pos,
                            X_sort[0:index, :], y_sort[0:index],
                            X_sort[index::, :], y_sort[index::])

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