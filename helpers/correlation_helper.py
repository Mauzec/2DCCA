import numpy as np

class Correlation:
    @staticmethod
    def distantion(X: np.ndarray, Y: np.ndarray) -> float:
        """
        Calculate the distance between two arrays.

        Parameters:
        X (np.ndarray): The first array.
        Y (np.ndarray): The second array.

        Returns:
        float: The distance between the two arrays.
        """
        return np.sum(np.abs(X - Y))
    
    @staticmethod
    def average(matrix1: np.ndarray, matrix2: np.ndarray) -> float:
        """
        Calculate the average correlation coefficient between two matrices.

        Parameters:
        matrix1 (np.ndarray): The first matrix.
        matrix2 (np.ndarray): The second matrix.

        Returns:
        float: The average correlation coefficient between the two matrices.
        """
        return np.corrcoef(matrix1.flatten(), matrix2.flatten())[0, 1]

    @staticmethod
    def cov(matrix1: np.ndarray, matrix2: np.ndarray) -> float:
        """
        Calculate the covariance between two matrices.

        Parameters:
        matrix1 (np.ndarray): The first matrix.
        matrix2 (np.ndarray): The second matrix.

        Returns:
        float: The covariance between the two matrices.
        """
        cov_matrix = np.cov(matrix1.flatten(), matrix2.flatten())
        std_matrix = np.sqrt(np.diag(cov_matrix))
        corr_matrix = cov_matrix / np.dot(std_matrix, std_matrix)
        return corr_matrix[0, 1]

    @staticmethod
    def pirson(matrix1: np.ndarray, matrix2: np.ndarray) -> float:
        """
        Calculate the Pearson correlation coefficient between two matrices.

        Parameters:
        matrix1 (np.ndarray): The first input matrix.
        matrix2 (np.ndarray): The second input matrix.

        Returns:
        float: The Pearson correlation coefficient between the two matrices.
        """
        vector1, vector2 = matrix1.flatten(), matrix2.flatten()
        covariance = np.cov(vector1, vector2)[0][1]
        return covariance / (np.std(vector1) * np.std(vector2))

    @staticmethod
    def cov_set(X: np.ndarray, Y: np.ndarray) -> float:
        """
        Calculate the average covariance between two sets of variables.

        Parameters:
        X (np.ndarray): The first set of variables.
        Y (np.ndarray): The second set of variables.

        Returns:
        float: The average covariance between X and Y.
        """
        return np.mean([Correlation.cov(x, y) for x, y in zip(X, Y)])

    @staticmethod
    def cov_element(matrix: np.ndarray) -> float:
        """
        Calculates the average correlation coefficient between all pairs of columns in a matrix.

        Parameters:
        matrix (np.ndarray): The input matrix.

        Returns:
        float: The average correlation coefficient between all pairs of columns.
        """
        correlations = np.corrcoef(matrix, rowvar=False)
        num_pairs = matrix.shape[1] * (matrix.shape[1] - 1) / 2
        return np.sum(correlations) / num_pairs
