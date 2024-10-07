import numpy as np
from abc import ABC, abstractmethod
import argparse
from collections import Counter


class NearestNeighborClassifier(ABC):
    """
    A Base nearest neighbor classifier class for implementing other classifiers .
    """

    def __init__(self, ord=2):
        """
        Nearest neighbor Constructor.

        :param ord: specify the order of the norm used in distance calculations.
                        The default value is 2, which corresponds to the Euclidean distance (optional).
        """

        self.X = None
        self.y = None
        self.ord = ord

    def fit(self, X, y):
        """
        Initialize the models using these data.

        :param X: Numpy array with shape (num_samples, num_features)
        :param y: Numpy integer array with length num_samples
        """
        self.X = X
        self.y = y

    @abstractmethod
    def predict(self, X_test):
        """
        Predict function for other implementations

        :param X:  Numpy array with shape (num_samples, num_features)
        :return: A length num_samples numpy array containing predicted labels.
        """

        raise Exception("Not implemented yet")


class KNNClassifier(NearestNeighborClassifier):
    """
    A K-nearest neighbor classifier class that classifies inputs based on the top K nearest neighbors .
    """

    def __init__(self, k=3, ord=2):
        """
        K-nearest neighbor Constructor.

        :param k: represents the number of nearest neighbors to consider when making
                        predictions, defaults to 3 (optional)
        :param ord: specify the order of the norm used in distance calculations.
                        The default value is 2, which corresponds to the Euclidean distance (optional).
        """
        super().__init__(ord)
        self.k = k
        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X_test):
        """
        Predict labels for a data set by calculating the distance between each input
        and samples each input, using the majority label as the prediction.

        :param X:  Numpy array with shape (num_samples, num_features)
        :return: A length num_samples numpy array containing predicted labels.

        Hint: to implement the KNN, function has to follow these steps:
             1. Iterate through each input sample.
             2. Calculate the distance between the input samples and the training sample.
             3. Select the top K samples with the smallest distance to the input.
             4. Get the majority class of the selected samples
        """

        if self.X is None or self.y is None:
            raise ValueError("Model is not fitted. Call 'fit' before 'predict'.")

        ### YOUR CODE HERE
        predictions = []
        for x in X_test:
            distances = np.linalg.norm(self.X - x, axis = 1)
            nearest_neighbors_index = np.argsort(distances)[:self.k]
            nearest_neighbors_label = self.y[nearest_neighbors_index]
            majority_label = Counter(nearest_neighbors_label).most_common(1)[0][0]
            predictions.append(majority_label)
        
        return np.array(predictions)


class EpsilonballNearestNeighborClassifier(NearestNeighborClassifier):
    """
    A Epsilon Ball-nearest neighbor classifier class that classifies inputs based on the top K nearest neighbors .
    """

    def __init__(self, epsilon=3, ord=2):
        """
        Epsilon Ball-nearest neighbor Constructor.

        :param epsilon:determines the margin of error allowed for a data point to be
                        considered. The default value is 3, defaults to 3 (optional)
        :param ord: specify the order of the norm used in distance calculations.
                        The default value is 2, which corresponds to the Euclidean distance (optional).
        """

        super().__init__(ord)
        self.epsilon = epsilon
        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X_test):
        """
        Predict labels for a data set by calculating the distance between each input
        and samples each input, using the majority label as the prediction.

        :param X:  Numpy array with shape (num_samples, num_features)
        :return: A length num_samples numpy array containing predicted labels.

        Hint: to implement the Epsilon-NN, function has to follow these steps:
             1. Iterate through each input sample.
             2. Calculate the distance between the input samples and the training sample.
             3. Select the samples that is under the threshold.
             4. Get the majority class of the selected samples
        """

        if self.X is None or self.y is None:
            raise ValueError(
                "Model is not fitted. Call 'fit' before 'predict_epsilon_ball'."
            )

        ### YOUR CODE HERE.
        predictions = []
        for x in X_test:
            distances = np.linalg.norm(self.X - x, ord = self.ord, axis = 1)
            inside_epsilon_ball = distances <= self.epsilon
            if np.any(inside_epsilon_ball):
                # print('no point inside')
                inside_epsilon_ball_labels = self.y[inside_epsilon_ball]
                majority_label = Counter(inside_epsilon_ball_labels).most_common(1)[0][0]
            else:
                nearest_neighbor_index = np.argmin(distances)
                majority_label = self.y[nearest_neighbor_index]
            predictions.append(majority_label)
        
        return np.array(predictions)

def knn_demo():
    # Create a simple dataset
    X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    y_train = np.array([0, 1, 0, 1, 0])
    X_test = np.array([[2, 2], [4, 4]])
    # Initialize and fit the k-NN classifier
    k = 3
    try:
        knn = KNNClassifier(k)
        knn.fit(X_train, y_train)

        # Predict the labels of the test data
        predictions = knn.predict(X_test)
        print("Predictions:", predictions)
    except Exception as e:
        print(f"Implementation gives error{e}")
    if (predictions == [0, 1]).all():
        print(
            "the result seems to be correct, you can use the test function to validate your implementation further."
        )


def epsilon_demo():
    # Create a simple dataset
    X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    y_train = np.array([0, 1, 0, 1, 0])
    X_test = np.array([[2, 2], [2, 4]])
    result_dict = {"ebnn": [0, 0], "knn": [0, 1]}
    # Initialize and fit the k-NN classifier
    epsilon = 3
    ebnn = EpsilonballNearestNeighborClassifier(epsilon, 1)
    ebnn.fit(X_train, y_train)

    # Predict the labels of the test data
    predictions = ebnn.predict(X_test)
    print("Predictions:", predictions)
    if (predictions == [0, 0]).all():
        print(
            "the result seems to be correct, you can use the test function to validate your implementation further."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a small demo to see if your code has any errors."
    )
    parser.add_argument(
        "--demo",
        metavar="N",
        type=str,
        help='Select a model to demo, "knn" for K-nearest neighbors, or "ebnn" for epsilon-ball nearest neighbors.',
        default="knn",
    )

    args = parser.parse_args()
    if args.demo == "knn":
        knn_demo()
    elif args.demo == "ebnn":
        epsilon_demo()
    else:
        raise Exception('Please pick between "knn" mode and "ebnn" mode.')
