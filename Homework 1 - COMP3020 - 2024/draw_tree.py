

from matplotlib import pyplot as plt
import numpy as np

def draw_tree(X, y, tree):
    """Plot training data and node boundaries for a 2d decision tree with
    at most three classes.

    In order for this code to work, the provided tree must have:
        - A _root attribute containing a root node
        - a get_depth() method

    Nodes must have:
        - left and right attributes
        - a split attribute containing a Split object

    """
    shapes = ['r*', 'bo', 'gs']
    for i, cl in enumerate(sorted(tree.classes)):
        indices = y == cl
        plt.plot(X[indices,0], X[indices,1], shapes[i], markersize=5)

    range_x_0 = np.max(X[:, 0]) - np.min(X[:, 0])
    range_x_1 = np.max(X[:, 1]) - np.min(X[:, 1])
    bounds = [np.min(X[:, 0]) - .05 * range_x_0,
              np.max(X[:, 0]) + .05 * range_x_0,
              np.min(X[:, 1]) - .05 * range_x_1,
              np.max(X[:, 1]) + .05 * range_x_1,]

    _draw_tree(tree._root, bounds, 0, tree.get_depth())
    plt.xlabel("$x_0$")
    plt.ylabel("$x_1$")
    plt.axis(bounds)
    plt.show()

def _draw_tree(node, bounds, cur_depth, max_depth):
    """ Recursive helper method. """
    linewidth = ((max_depth - cur_depth) / max_depth) * 3 + .1

    if node.left is not None:
        if node.split.dim == 0:
            plt.plot([node.split.pos, node.split.pos],
                     [bounds[2], bounds[3]], linewidth=linewidth,
                     color='black')

            left_bounds = [bounds[0], node.split.pos, bounds[2],
                           bounds[3]]
            _draw_tree(node.left, left_bounds, cur_depth + 1, max_depth)
            right_bounds = [node.split.pos, bounds[1],
                            bounds[2], bounds[3]]
            _draw_tree(node.right, right_bounds, cur_depth + 1, max_depth)


        if node.split.dim == 1:
            plt.plot([bounds[0], bounds[1]],
                     [node.split.pos, node.split.pos], linewidth=linewidth,
                     color='black')
            left_bounds = [bounds[0], bounds[1], bounds[2],
                           node.split.pos]
            _draw_tree(node.left, left_bounds, cur_depth + 1, max_depth)
            right_bounds = [bounds[0], bounds[1], node.split.pos,
                            bounds[3]]
            _draw_tree(node.right, right_bounds, cur_depth + 1, max_depth)