import unittest
import numpy as np
import time
import sklearn.datasets
from k_nearest_neighbors import KNNClassifier,EpsilonballNearestNeighborClassifier

class KNNTestCase(unittest.TestCase):

    def setUp(self):
        # These data sets are carefully selected so that there should be
        # no ties during classifier construction.  This means that there should
        # be a unique correct classifier for each.

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

    def K_1_majority_helper(self, X, y, expected_class):
        classsifier = KNNClassifier(k=1)
        classsifier.fit(X, y)
        X_test = np.random.random((100, X.shape[1])) - 1 * 2
        y_test = classsifier.predict(X_test)
        self.assertTrue(np.alltrue(y_test == expected_class))

    def test_k1_classsifier(self):
        self.K_1_majority_helper(self.X2, self.y2, 0)
        self.K_1_majority_helper(self.X2_big, self.y2_big, 1)
        self.K_1_majority_helper(self.X3, self.y3, 4)
    def k_3_training_helper(self, X, y, noise=0):
        classifier = KNNClassifier(k=1)
        classifier.fit(X, y)
        y_test = classifier.predict(X + (np.random.random(X.shape) - .5) * noise)
        np.testing.assert_array_equal(y, y_test)

    def test_k_3_on_training_points(self):
        self.k_3_training_helper(self.X2, self.y2)
        self.k_3_training_helper(self.X2_big, self.y2_big)
        self.k_3_training_helper(self.X2_3classes, self.y2_3classes)
        self.k_3_training_helper(self.X3, self.y3)
        
    def test_k_3_on_perturbed_training_points(self):
        # Since the training data is all rounded to 2 decimal places, this
        # amount of perturbation shouldn't be able to push us across a
        # split boundary.
        self.k_3_training_helper(self.X2, self.y2, .009)
        self.k_3_training_helper(self.X2_big, self.y2_big, .009)
        self.k_3_training_helper(self.X2_3classes, self.y2_3classes,
                                           .009)
        self.k_3_training_helper(self.X3, self.y3, .009)
        
        
    def test_k_2_tree_predictions(self):
        classifier = KNNClassifier(k=2)
        classifier.fit(self.X2_big, self.y2_big)
        y_test = classifier.predict(np.array([[.6, .6],
                                        [.4, .3],
                                        [.2, .4],
                                        [.6, .2]]))

        np.testing.assert_array_equal(np.array([1, -1, 1, -1]), y_test)
        

class EpsilonBallTestCase(unittest.TestCase):

    def setUp(self):
        # These data sets are carefully selected so that there should be
        # no ties during classifier construction.  This means that there should
        # be a unique correct classifier for each.

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

    def Ep_max_majority_helper(self, X, y, expected_class):
        classsifier = EpsilonballNearestNeighborClassifier(epsilon=100)
        classsifier.fit(X, y)
        X_test = np.random.random((100, X.shape[1])) - 1 * 2
        y_test = classsifier.predict(X_test)
        self.assertTrue(np.alltrue(y_test == expected_class))

    def test_max_classsifier(self):
        self.Ep_max_majority_helper(self.X2, self.y2, 0)
        self.Ep_max_majority_helper(self.X2_big, self.y2_big, -1)
        self.Ep_max_majority_helper(self.X3, self.y3, 0)
        
    def Ep_3_training_helper(self, X, y, noise=0):
        classifier = EpsilonballNearestNeighborClassifier(epsilon=3)
        classifier.fit(X, y)
        y_test = classifier.predict(X + (np.random.random(X.shape) - .5) * noise)
        np.testing.assert_array_equal(y, y_test)
    
    def Ep_0_1_training_helper(self, X, y, noise=0):
        classifier = EpsilonballNearestNeighborClassifier(epsilon=0.1)
        classifier.fit(X, y)
        y_test = classifier.predict(X + (np.random.random(X.shape) - .5) * noise)
        np.testing.assert_array_equal(y, y_test)
    
    def test_on_training_points(self):
        self.Ep_0_1_training_helper(self.X2, self.y2)
        self.Ep_0_1_training_helper(self.X2_3classes, self.y2_3classes)
        self.Ep_0_1_training_helper(self.X2_big, self.y2_big)
        
        self.Ep_3_training_helper(self.X3, self.y3)
        # Since the training data is all rounded to 2 decimal places, this
    def test_e_3_on_perturbed_training_points(self):
        # Since the training data is all rounded to 2 decimal places, this
        # amount of perturbation shouldn't be able to push us across a
        # split boundary.
        self.Ep_0_1_training_helper(self.X2_3classes, self.y2_3classes,
                                           .009)
        self.Ep_3_training_helper(self.X3, self.y3, .009)
        
        
    def test_ep_0_5_predictions(self):
        classifier = EpsilonballNearestNeighborClassifier(epsilon=0.5)
        classifier.fit(self.X2_big, self.y2_big)
        y_test = classifier.predict(np.array([[.6, .6],
                                        [.4, .3],
                                        [.2, .4],
                                        [.6, .2]]))

        np.testing.assert_array_equal(np.array([-1, 1, 1, -1]), y_test)
    
        
    def test_ep_0_5_ord_1_predictions(self):
        classifier = EpsilonballNearestNeighborClassifier(epsilon=0.5,ord=1)
        classifier.fit(self.X2_big, self.y2_big)
        y_test = classifier.predict(np.array([[.6, .6],
                                        [.4, .3],
                                        [.2, .4],
                                        [.6, .2]]))

        np.testing.assert_array_equal(np.array([-1, -1, 1, -1]), y_test)


if __name__ == '__main__':
    unittest.main()
