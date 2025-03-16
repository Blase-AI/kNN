import numpy as np
from typing import Optional, List
        
class KNearestNeighbor:
    """ a kNN classifier with customizable distance metrics """

    def __init__(self) -> None:
        pass

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X: np.ndarray, k: int = 1, num_loops: int = 0, distance_metric: str = 'l2') -> np.ndarray:
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        - num_loops: Determines which implementation to use to compute distances
          between training points and testing points.
        - distance_metric: The distance metric to use. Options: 'l1', 'l2', 'cosine', 'chebyshev'.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X, distance_metric)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X, distance_metric)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X, distance_metric)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)

        return self.predict_labels(dists, k=k)

    def choose_best_k(self, X: np.ndarray, y: np.ndarray, k_values: Optional[List[int]] = None, verbose: bool = True) -> int:
        """
        Automatically selects the best k value by performing cross-validation.

        This method tests different values of k and calculates the accuracy 
        of the classifier on the validation set, returning the k that gives the
        highest accuracy. It also prints the accuracy for each tested k.

        Inputs:
        - X: A numpy array of shape (num_val, D) containing the validation data.
        - y: A numpy array of shape (num_val,) containing the validation labels.
        - k_values: (Optional) A list of possible k values to test. If None, defaults to [1, 3, 5, 7, 9, 11].
        - verbose: If True, prints accuracy for each k.

        Returns:
        - best_k: The value of k that achieved the highest accuracy on the validation set.
        """
        if k_values is None:
            k_values = [1, 3, 5, 7, 9, 11]
    
        if not isinstance(k_values, (list, np.ndarray)) or len(k_values) == 0:
            raise ValueError("k_values must be a non-empty list or numpy array of positive integers.")

        k_values = [k for k in k_values if isinstance(k, int) and k > 0]
        if len(k_values) == 0:
            raise ValueError("k_values must contain at least one valid positive integer.")

        best_k = k_values[0]
        best_accuracy = 0
        accuracies = {}

        for k in k_values:
            y_pred = self.predict(X, k=k)
            accuracy = np.mean(y_pred == y)
            accuracies[k] = accuracy

            if verbose:
                print(f"k = {k}, Accuracy = {accuracy:.4f}")

            if accuracy > best_accuracy:
                best_k = k
                best_accuracy = accuracy

        if verbose:
            print(f"\nBest k: {best_k} with Accuracy: {best_accuracy:.4f}")

        return best_k



    def compute_distances_two_loops(self, X: np.ndarray, distance_metric: str = 'l2') -> np.ndarray:
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the
        test data.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.
        - distance_metric: The distance metric to use. Options: 'l1', 'l2', 'cosine', 'chebyshev'.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the distance between the ith test point and the jth training point.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                dists[i][j] = self._compute_distance(X[i], self.X_train[j], distance_metric)
        return dists

    def compute_distances_one_loop(self, X: np.ndarray, distance_metric: str = 'l2') -> np.ndarray:
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            dists[i, :] = self._compute_distance_vectorized(X[i], self.X_train, distance_metric)
        return dists

    def compute_distances_no_loops(self, X: np.ndarray, distance_metric: str = 'l2') -> np.ndarray:
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Input / Output: Same as compute_distances_two_loops
        """
        if distance_metric == 'l2':
            num_test = X.shape[0]
            num_train = self.X_train.shape[0]
            test_sq = np.sum(X ** 2, axis=1).reshape(-1, 1)
            train_sq = np.sum(self.X_train ** 2, axis=1)
            dot_p = X @ self.X_train.T
            dists = np.sqrt(test_sq + train_sq - 2 * dot_p)
        elif distance_metric == 'cosine':
            norm_X = np.linalg.norm(X, axis=1).reshape(-1, 1)
            norm_X_train = np.linalg.norm(self.X_train, axis=1)
            dot_p = X @ self.X_train.T
            dists = 1 - (dot_p / (norm_X * norm_X_train + 1e-10)) 
        else:
            num_test = X.shape[0]
            num_train = self.X_train.shape[0]
            dists = np.zeros((num_test, num_train))
            for i in range(num_test):
                dists[i, :] = self._compute_distance_vectorized(X[i], self.X_train, distance_metric)
        return dists

    def _compute_distance(self, x1: np.ndarray, x2: np.ndarray, distance_metric: str = 'l2') -> float:
        """
        Compute the distance between two vectors x1 and x2.

        Inputs:
        - x1, x2: Two vectors of the same dimension.
        - distance_metric: The distance metric to use. Options: 'l1', 'l2', 'cosine', 'chebyshev'.

        Returns:
        - distance: The computed distance.
        """
        if distance_metric == 'l1':
            return np.sum(np.abs(x1 - x2)) 
        elif distance_metric == 'l2':
            return np.sqrt(np.sum((x1 - x2) ** 2))  
        elif distance_metric == 'cosine':
            norm_x1 = np.linalg.norm(x1)
            norm_x2 = np.linalg.norm(x2)
            if norm_x1 == 0 or norm_x2 == 0:
                return 1  
            return 1 - np.dot(x1, x2) / (norm_x1 * norm_x2)  
        elif distance_metric == 'chebyshev':
            return np.max(np.abs(x1 - x2))  
        else:
            raise ValueError(f"Unknown distance metric: {distance_metric}")

    def _compute_distance_vectorized(self, x: np.ndarray, X_train: np.ndarray, distance_metric: str = 'l2') -> np.ndarray:
        """
        Compute the distance between a vector x and each row in X_train.

        Inputs:
        - x: A single vector.
        - X_train: A numpy array of shape (num_train, D) containing training data.
        - distance_metric: The distance metric to use. Options: 'l1', 'l2', 'cosine', 'chebyshev'.

        Returns:
        - distances: A numpy array of shape (num_train,) containing distances.
        """
        if distance_metric == 'l1':
            return np.sum(np.abs(X_train - x), axis=1)   
        elif distance_metric == 'l2':
            return np.sqrt(np.sum((X_train - x) ** 2, axis=1)) 
        elif distance_metric == 'cosine':
            norm_x = np.linalg.norm(x)
            norm_X_train = np.linalg.norm(X_train, axis=1)
            dot_p = np.dot(X_train, x)
            return 1 - (dot_p / (norm_X_train * norm_x + 1e-10)) 
        elif distance_metric == 'chebyshev':
            return np.max(np.abs(X_train - x), axis=1) 
        else:
            raise ValueError(f"Unknown distance metric: {distance_metric}")

    def predict_labels(self, dists: np.ndarray, k: int = 1) -> np.ndarray:
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance between the ith test point and the jth training point.
        - k: The number of nearest neighbors to consider.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            closest_y = np.argsort(dists[i])[:k]
            closest_y = self.y_train[closest_y]
            class_values, class_freq = np.unique(closest_y, return_counts=True)
            final_label = class_values[np.argmax(class_freq)]
            y_pred[i] = final_label

        return y_pred
    

 