import cv2
import numpy as np
from abstract.classifier import ClassifierAlgorithm

class TDPLSC(ClassifierAlgorithm):
    def __init__(self, dimension: int = 10, distance_function = lambda x, y: np.linalg.norm(x - y), is_max: bool = True) -> None:
        '''
        Initialize a TDPLS_Cascade object.
        
        Parameters:
        - dimension (int): The dimension of the TDPLS_Cascade object. Default is 10.
        - distance_function (function): The distance function used to calculate the distance between two vectors. Default is Euclidean distance.
        - is_max (bool): A flag indicating whether to maximize or minimize the objective function. Default is True (maximize).
        '''
        
        self.U = None
        self.V = None
        self.W = {"x": [None, None], "y": [None, None]}
        self.X_c = None
        self.Y_c = None
        self.links_x = None
        self.links_y = None
        self.dimension = dimension
        self.distance_function = distance_function
        self.regular_s: float = 5 * 10**(-4)
        self.is_max = is_max
        
        assert dimension > 0, "Dimension must be greater than 0"
        assert self.regular_s > 0, "Regularization parameter S must be greater than 0"
        assert callable(distance_function), "Distance function must be callable"
        assert isinstance(is_max, bool), "is_max must be a boolean"
        assert isinstance(dimension, int), "Dimension must be an integer"
    
    def fit(self, X: list, Y: list, with_RRPP: bool = False) -> None:
        self.X_c = np.mean(X, axis=0)
        self.Y_c = np.mean(Y, axis=0)
        
        X -= self.X_c
        Y -= self.Y_c    
        assert len(X) == len(Y), "The number of elements in X and Y must be equal"
        assert len(X) > 0, "The number of elements in X must be greater than 0"
        assert len(Y) > 0, "The number of elements in Y must be greater than 0"    
        
        self.W["x"][0], self.W["y"][0] = self._calculate_projections(X, Y, rows=True, with_RRPP=with_RRPP)
        
        self.U = [self.W['x'][0].T @ x for x in X]
        self.V = [self.W["y"][0].T @ y for y in Y]
        
        self.W["x"][1], self.W["y"][1] = self._calculate_projections(X, Y, rows=False, with_RRPP=with_RRPP)
        
        self.U = np.array([u @ self.W["x"][1]for u in self.U])
        self.V = np.array([v @ self.W["y"][1] for v in self.V])
        
        assert self.W["x"][0].shape == self.W["y"][0].shape, "The dimensions of the matrices must be equal"
        assert self.W["x"][1].shape == self.W["y"][1].shape, "The dimensions of the matrices must be equal"
        assert self.U.shape[0] == len(X), "The number of elements in U must be equal to the number of elements in X"
        assert self.V.shape[0] == len(Y), "The number of elements in V must be equal to the number of elements in Y"
        
        
    def transform(self, matrix_set: np.ndarray, is_x: bool=True) -> np.ndarray:
        W = self.W["x"] if is_x else self.W["y"]
        X_c = self.X_c if is_x else self.Y_c
        return np.array([W[0].T @ (m - X_c) @ W[1] for m in matrix_set])
        
    def predict(self, matrixes: list, is_x: bool = True) -> tuple[int, float]:
        assert self.U is not None, "The model must be fitted before predicting"
        assert self.V is not None, "The model must be fitted before predicting"
        assert len(matrixes) > 0, "The number of elements in matrices must be greater than 0"
        assert isinstance(is_x, bool), "is_x must be a boolean"
        
        assert self.U is not None, "The model must be fitted before predicting"
        assert self.V is not None, "The model must be fitted before predicting"
        assert len(matrixes) > 0, "The number of elements in matrices must be greater than 0"
        assert isinstance(is_x, bool), "is_x must be a boolean"
        transformed_matrices = self.transform(matrixes, is_x=is_x)
        training_matrices = self.U if is_x else self.V

        distance_matrix = np.zeros((len(matrixes), len(training_matrices)), dtype="float64")

        for udx, transformed_matrix in enumerate(transformed_matrices):
            for vdx, training_matrix in enumerate(training_matrices):
                distance = self.distance_function(transformed_matrix, training_matrix)
                distance_matrix[udx][vdx] = distance
        assert distance_matrix.shape[0] == len(matrixes), "The number of elements in the distance matrix must be equal to the number of elements in matrices"
        results = [[vector.argmax(), vector.max()] if self.is_max else [vector.argmin(), vector.min()] for vector in distance_matrix]

        return results  

    def _calculate_projections(self, X: np.ndarray, Y: np.ndarray, rows: bool = True,
                     with_RRPP: bool = False) -> tuple[np.ndarray, np.ndarray]: 
        C_xy = self._calculate_covariance_matrix(X, Y, rows=rows)
        C_yx = self._calculate_covariance_matrix(Y, X, rows=rows)
         
        S_x = C_xy @ C_yx
        S_y = C_yx @ C_xy 
        
        S_x = S_x + self.regular_s * np.identity(S_x.shape[0])
        S_y = S_y + self.regular_s * np.identity(S_y.shape[0]) 
        
        l_x, v_x = np.linalg.eig(S_x)
        l_y, v_y = np.linalg.eig(S_y)
        
        if with_RRPP:
            v_x = v_x.T[:, l_x.argsort()[-self.dimension:][::-1]]
            v_y = v_y.T[:, l_y.argsort()[-self.dimension:][::-1]]
        else:
            v_x = v_x[:, l_x.argsort()[::-1]].T
            v_y = v_y[:, l_y.argsort()[::-1]].T
            
        return v_x, v_y            
            
    def _calculate_covariance_matrix(self, matrixes_1: np.ndarray, matrixes_2: np.ndarray, 
                                     rows: bool = True) -> np.ndarray:
        C = np.sum([matrixes_1[i] @ matrixes_2[i].T if rows else matrixes_1[i].T @ matrixes_2[i] for i in range(len(matrixes_1))], axis=0)
        return C

    