import unittest
import numpy as np
import time
import sklearn.datasets
from decision_tree import DecisionTree


class DecisionTreeTestCaseGini(unittest.TestCase):

    def setUp(self):
        # These data sets are carefully selected so that there should be
        # no ties during tree construction.  This means that there should
        # be a unique correct tree for each.

        self.X2 = np.array([[0.88, 0.39],
                            [0.49, 0.52],
                            [0.68, 0.26],
                            [0.57, 0.51],
                            [0.61, 0.73]])
        self.y2 = np.array([1, 0, 0, 0, 1])

        self.X2_big = np.array([[0.41, 0.17],
                                [0.45, 0.29],
                                [0.96, 0.46],
                                [0.67, 0.19],
                                [0.76, 0.2],
                                [0.75, 0.59],
                                [0.24, 0.1],
                                [0.82, 0.79],
                                [0.08, 0.16],
                                [0.62, 0.44],
                                [0.22, 0.74],
                                [0.5, 0.48]])

        self.y2_big = np.array([-1, -1, -1, -1, -1, 1, 1, -1, 1, 1, 1, -1])

        self.X3 = np.array([[-2.4, -1.7, 1.2],
                            [-3.6, 4.7, 0.2],
                            [1.9, 2., -1.5],
                            [1.4, -0.9, -0.6],
                            [4.8, -0.7, -1.8],
                            [-1.4, 4.3, -4.9],
                            [-4.7, -2.7, 2.4],
                            [-4., 3.7, -2.7],
                            [-1.6, 3.7, 2.6],
                            [-1.5, -3.1, -0.9],
                            [-2.4, -4.7, 0.6],
                            [4.3, 0.2, 2.]])

        self.y3 = np.array([4, 0, 0, 0, 4, 0, 4, 4, 0, 4, 0, 0])

        self.X2_3classes = np.array([[0.72, 0.16],
                                     [0.18, 0.37],
                                     [0.02, 0.53],
                                     [0.97, 0.26],
                                     [0.38, 0.],
                                     [0.61, 0.71],
                                     [0.53, 0.2],
                                     [0.66, 0.42],
                                     [0.78, 0.88],
                                     [0.79, 0.26]])
        self.y2_3classes = np.array([0, 2, 2, 2, 0, 1, 0, 0, 1, 0])

    def depth0_majority_helper(self, X, y, expected_class):
        tree = DecisionTree(max_depth=0)
        tree.fit(X, y)
        X_test = np.random.random((100, 2)) - .5 * 2
        y_test = tree.predict(X_test)
        self.assertTrue(np.alltrue(y_test == expected_class))

    def test_depth0_trees(self):
        self.depth0_majority_helper(self.X2, self.y2, 0)
        self.depth0_majority_helper(self.X2_big, self.y2_big, -1)
        self.depth0_majority_helper(self.X3, self.y3, 0)
        self.depth0_majority_helper(self.X2_3classes, self.y2_3classes, 0)
        
    def test_depth_2_tree_predictions(self):
        tree = DecisionTree(max_depth=2)
        tree.fit(self.X2_big, self.y2_big)
        y_test = tree.predict(np.array([[.6, .6],
                                        [.4, .8],
                                        [.2, .4],
                                        [.6, .2]]))

        np.testing.assert_array_equal(np.array([-1, -1, 1, -1]), y_test)
        
    def test_classes_attribute(self):
        tree = DecisionTree()
        tree.fit(self.X2, self.y2)
        self.assertEqual(tree.classes, set(self.y2))

        tree = DecisionTree()
        tree.fit(self.X3, self.y3)
        self.assertEqual(tree.classes, set(self.y3))

    def tree_depth_helper(self, X, y, max_depth, expected_depth):
        tree = DecisionTree(max_depth=max_depth)
        tree.fit(X, y)
        self.assertEqual(expected_depth, tree.get_depth())

    def test_get_depth_zero(self):
        self.tree_depth_helper(self.X2, self.y2, 0, 0)
        self.tree_depth_helper(self.X2_big, self.y2_big, 0, 0)
        self.tree_depth_helper(self.X2_3classes, self.y2_3classes, 0, 0)
        self.tree_depth_helper(self.X3, self.y3, 0, 0)

    def test_get_depth_one(self):
        self.tree_depth_helper(self.X2, self.y2, 1, 1)
        self.tree_depth_helper(self.X2_big, self.y2_big, 1, 1)
        self.tree_depth_helper(self.X2_3classes, self.y2_3classes, 1, 1)
        self.tree_depth_helper(self.X3, self.y3, 1, 1)


class DecisionTreeTestCaseIG(unittest.TestCase):

    def setUp(self):
        # These data sets are carefully selected so that there should be
        # no ties during tree construction.  This means that there should
        # be a unique correct tree for each.

        self.X2 = np.array([[0.88, 0.39],
                            [0.49, 0.52],
                            [0.68, 0.26],
                            [0.57, 0.51],
                            [0.61, 0.73]])
        self.y2 = np.array([1, 0, 0, 0, 1])

        self.X2_big = np.array([[0.41, 0.17],
                                [0.45, 0.29],
                                [0.96, 0.46],
                                [0.67, 0.19],
                                [0.76, 0.2],
                                [0.75, 0.59],
                                [0.24, 0.1],
                                [0.82, 0.79],
                                [0.08, 0.16],
                                [0.62, 0.44],
                                [0.22, 0.74],
                                [0.5, 0.48]])

        self.y2_big = np.array([-1, -1, -1, -1, -1, 1, 1, -1, 1, 1, 1, -1])

        self.X3 = np.array([[-2.4, -1.7, 1.2],
                            [-3.6, 4.7, 0.2],
                            [1.9, 2., -1.5],
                            [1.4, -0.9, -0.6],
                            [4.8, -0.7, -1.8],
                            [-1.4, 4.3, -4.9],
                            [-4.7, -2.7, 2.4],
                            [-4., 3.7, -2.7],
                            [-1.6, 3.7, 2.6],
                            [-1.5, -3.1, -0.9],
                            [-2.4, -4.7, 0.6],
                            [4.3, 0.2, 2.]])

        self.y3 = np.array([4, 0, 0, 0, 4, 0, 4, 4, 0, 4, 0, 0])

        self.X2_3classes = np.array([[0.72, 0.16],
                                     [0.18, 0.37],
                                     [0.02, 0.53],
                                     [0.97, 0.26],
                                     [0.38, 0.],
                                     [0.61, 0.71],
                                     [0.53, 0.2],
                                     [0.66, 0.42],
                                     [0.78, 0.88],
                                     [0.79, 0.26]])
        self.y2_3classes = np.array([0, 2, 2, 2, 0, 1, 0, 0, 1, 0])

    def depth0_majority_helper(self, X, y, expected_class):
        tree = DecisionTree(max_depth=0,mode="ig")
        tree.fit(X, y)
        X_test = np.random.random((100, 2)) - .5 * 2
        y_test = tree.predict(X_test)
        self.assertTrue(np.alltrue(y_test == expected_class))

    def test_depth0_trees(self):
        self.depth0_majority_helper(self.X2, self.y2, 0)
        self.depth0_majority_helper(self.X2_big, self.y2_big, -1)
        self.depth0_majority_helper(self.X3, self.y3, 0)
        self.depth0_majority_helper(self.X2_3classes, self.y2_3classes, 0)
        
    def test_depth_2_tree_predictions(self):
        tree = DecisionTree(max_depth=2,mode="ig")
        tree.fit(self.X2_big, self.y2_big)
        y_test = tree.predict(np.array([[.6, .6],
                                        [.4, .8],
                                        [.2, .4],
                                        [.6, .2]]))

        np.testing.assert_array_equal(np.array([-1, -1, 1, -1]), y_test)
        
    def test_classes_attribute(self):
        tree = DecisionTree(mode="ig")
        tree.fit(self.X2, self.y2)
        self.assertEqual(tree.classes, set(self.y2))

        tree = DecisionTree(mode="ig")
        tree.fit(self.X3, self.y3)
        self.assertEqual(tree.classes, set(self.y3))

    def tree_depth_helper(self, X, y, max_depth, expected_depth):
        tree = DecisionTree(max_depth=max_depth,mode="ig")
        tree.fit(X, y)
        self.assertEqual(expected_depth, tree.get_depth())

    def test_get_depth_zero(self):
        self.tree_depth_helper(self.X2, self.y2, 0, 0)
        self.tree_depth_helper(self.X2_big, self.y2_big, 0, 0)
        self.tree_depth_helper(self.X2_3classes, self.y2_3classes, 0, 0)
        self.tree_depth_helper(self.X3, self.y3, 0, 0)

    def test_get_depth_one(self):
        self.tree_depth_helper(self.X2, self.y2, 1, 1)
        self.tree_depth_helper(self.X2_big, self.y2_big, 1, 1)
        self.tree_depth_helper(self.X2_3classes, self.y2_3classes, 1, 1)
        self.tree_depth_helper(self.X3, self.y3, 1, 1)



if __name__ == '__main__':
    unittest.main()