import cv2
import numpy as np
from threading import Thread
from abstract.classifier import ClassifierAlgorithm
from queue import Queue
import matplotlib.pyplot as plt

class TDCCAC(ClassifierAlgorithm):
    def __init__(self, dimension: int = 10, distance_function = lambda x, y: np.linalg.norm(x - y), is_max: bool = True) -> None:
        '''
        Initializes a new instance of the TDCCA class.
        
        dimension (int): The dimension of the canonical correlation space. Default is 10.
        distance_function (function): The distance function used to measure the similarity between samples. 
            Default is the Euclidean distance function.
        is_max (bool): Determines whether to maximize or minimize the canonical correlation. Default is True (maximize).
        
        Raises:
            AssertionError: If dimension is not greater than 0, regularization parameters are not greater than 0,
                distance_function is not callable, or is_max is not a boolean.
        '''
        
        self.U = None
        self.V = None
        self.W = {"x": [None, None], "y": [None, None]}
        self.mean_X = None
        self.mean_Y = None
        self.links_x = None
        self.links_y = None
        self.dimension = dimension
        self.distance_function = distance_function
        self.regular_C: float = 10**(-4)
        self.regular_S: float = 5 * 10**(-4)
        self.is_max = is_max
        
        assert dimension > 0, "Dimension must be greater than 0"
        assert self.regular_C > 0, "Regularization parameter C must be greater than 0"
        assert self.regular_S > 0, "Regularization parameter S must be greater than 0"
        assert callable(distance_function), "Distance function must be callable"
        assert isinstance(is_max, bool), "is_max must be a boolean"
        assert isinstance(dimension, int), "Dimension must be an integer"

    def fit(self, X: list, Y: list, with_RRPP: bool = False) -> None:
        self.mean_X = np.mean(X, axis=0)
        self.mean_Y = np.mean(Y, axis=0)
        X -= self.mean_X
        Y -= self.mean_Y    
        self.X = X
        self.Y = Y     
        assert len(X) == len(Y), "The number of elements in X and Y must be equal"
        assert len(X) > 0, "The number of elements in X must be greater than 0"
        assert len(Y) > 0, "The number of elements in Y must be greater than 0"
        
        self._calculate_weights(X, Y, with_RRPP)
        assert self.W["x"][0].shape == self.W["y"][0].shape, "The dimensions of the matrices must be equal"
        assert self.W["x"][1].shape == self.W["y"][1].shape, "The dimensions of the matrices must be equal"

        self.U = self.transform(X + self.mean_X, is_x=True)
        assert self.U.shape[0] == len(X), "The number of elements in U must be equal to the number of elements in X"
        self.V = self.transform(Y + self.mean_Y, is_x=False)
        assert self.V.shape[0] == len(Y), "The number of elements in V must be equal to the number of elements in Y"

    def _calculate_weights(self, X, Y, with_RRPP):
        self.W['x'][0], self.W['y'][0] = self._calculate_projections(X, Y, rows=True, with_RRPP=with_RRPP)
        self.U, self.V = [], []
        for i in range(len(X)):
            self.U.append(self.W["x"][0].T @ X[i])
            self.V.append(self.W["y"][0].T @ Y[i])
        
        self.W['x'][1], self.W['y'][1] = self._calculate_projections(X, Y, rows=False, with_RRPP=with_RRPP)
        for i in range(len(X)):
            self.U[i] = self.U[i] @ self.W["x"][1]
            self.V[i] = self.V[i] @ self.W["y"][1]
        

    def transform(self, matrix_set: np.ndarray, is_x: bool=True) -> np.ndarray:
        return np.array([self.W["x"][0].T @ (matrix - self.mean_X) @ self.W["x"][1] if is_x else self.W["y"][0].T @ (matrix - self.mean_X) @ self.W["y"][1] for matrix in matrix_set])

    def predict(self, matrices: list, is_x: bool = True) -> list[tuple[int, float]]:
        ''' Predict the output for the given input data.
            matrices: list - the input data.
            is_x: bool = True - if True, predict the output for the X data,
                                if False, predict the output for the Y data.
        '''
        
        assert self.U is not None, "The model must be fitted before predicting"
        assert self.V is not None, "The model must be fitted before predicting"
        assert len(matrices) > 0, "The number of elements in matrices must be greater than 0"
        assert isinstance(is_x, bool), "is_x must be a boolean"
        transformed_matrices = self.transform(matrices, is_x=is_x)
        training_matrices = self.U if is_x else self.V

        distance_matrix = np.zeros((len(matrices), len(training_matrices)), dtype="float64")

        for udx, transformed_matrix in enumerate(transformed_matrices):
            for vdx, training_matrix in enumerate(training_matrices):
                distance = self.distance_function(transformed_matrix, training_matrix)
                distance_matrix[udx][vdx] = distance
        assert distance_matrix.shape[0] == len(matrices), "The number of elements in the distance matrix must be equal to the number of elements in matrices"
        results = [[vector.argmax(), vector.max()] if self.is_max else [vector.argmin(), vector.min()] for vector in distance_matrix]

        return results  
    
    def _calculate_projections(self, X: np.ndarray, Y: np.ndarray, rows: bool = True,
                     with_RRPP: bool = False) -> tuple[np.ndarray, np.ndarray, bool]: 
        ''' Calculate the weights for the given data. 
            rows: bool = True - if True, calculate the weights by rows,
                                if False, calculate the weights by columns.
        '''
        C_xx = self._calculate_cov(X, X, rows=rows)
        C_yy = self._calculate_cov(Y, Y, rows=rows)
        C_xy = self._calculate_cov(X, Y, rows=rows)
        C_yx = self._calculate_cov(Y, X, rows=rows)
        assert C_xx.shape == C_yy.shape, "The dimensions of the matrices must be equal"
        assert C_xy.shape == C_yx.shape, "The dimensions of the matrices must be equal"
        
        C_xx = C_xx + self.regular_C * np.identity(C_xx.shape[0]) 
        C_yy = C_yy + self.regular_C * np.identity(C_yy.shape[0]) 
         
        S_x = np.linalg.inv(C_xx) @ C_xy @ np.linalg.inv(C_yy) @ C_yx
        S_y = np.linalg.inv(C_yy) @ C_yx @ np.linalg.inv(C_xx) @ C_xy 
        assert S_x.shape == S_y.shape, "The dimensions of the matrices must be equal"
        
        S_x = S_x + self.regular_S * np.identity(S_x.shape[0])
        S_y = S_y + self.regular_S * np.identity(S_y.shape[0]) 
        
        # Calculate the eigenvectors and eigenvalues
        l_x, v_x = np.linalg.eig(S_x)
        l_y, v_y = np.linalg.eig(S_y)
        assert l_x.shape == l_y.shape, "The dimensions of the matrices must be equal"
        
        # Sort the eigenvectors by the eigenvalues
        if with_RRPP:
            v_x = v_x.T[:, l_x.argsort()[-self.dimension:][::-1]]
            v_y = v_y.T[:, l_y.argsort()[-self.dimension:][::-1]]
        else:
            v_x = v_x[:, l_x.argsort()[::-1]].T
            v_y = v_y[:, l_y.argsort()[::-1]].T
        
        return v_x, v_y
               
            
    def _calculate_cov(self, matrixes_1: np.ndarray, matrixes_2: np.ndarray, 
                                     rows: bool = True) -> np.ndarray:
        ''' Calculate the covariance matrix.
            rows: bool = True - if True, calculate the covariance matrix by rows,
                                if False, calculate the covariance matrix by columns.
        '''
        
        if rows:
            C = np.sum([matrixes_1[i] @ matrixes_2[i].T for i in range(len(matrixes_1))], axis=0)
        else:
            C = np.sum([matrixes_1[i].T @ matrixes_2[i] for i in range(len(matrixes_1))], axis=0)
        return C
               