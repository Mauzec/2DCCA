from abc import ABCMeta, abstractmethod
import numpy as np
class ClassifierAlgorithm(metaclass=ABCMeta):
    """
    Abstract class for an classifier algorithm.
    """

    @abstractmethod
    def fit(self, input_data: list, target_data: list, with_rrpp: bool = False) -> None:
        """
        Fit the model to the data.
        """
        pass
    
    @abstractmethod
    def predict(self, input_data: list[str], target_data: list[str]) -> tuple[int, float]:
        """
        Predict the output for the given input data.
        """
        pass
    
    @abstractmethod
    def transform(self, matrix_set: np.ndarray, is_input_data: bool=True) -> np.ndarray:
        """
        Transform the data.
        """
        pass