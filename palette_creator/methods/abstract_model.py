from abc import ABC, abstractmethod
import numpy as np

class Model(ABC):

    @abstractmethod
    def create_palette(self, image: np.ndarray) -> tuple[list, list]:
        """Create the palette of an image sorted by the proportions.

        :param np.ndarray image: image in shape of (M, N, 3)
        :returns: a 2-tuple of the color palette (list of 3-tuple where a tuple means the rgb color) and the proportions  

        """
        
        pass
    
    @staticmethod
    def _validate_image(image: np.ndarray) -> None:
        """ Validate if the image has the correct shape (M, N, 3)

        :param np.ndarray image: numpy array with colors as numbers
        :returns: None

        """
        
        is_correct_shape = image.ndim == 3 and image.shape[-1] == 3
        is_numeric = np.issubdtype(image.dtype, np.number)

        if not is_correct_shape:
            raise IncorrectShapeError("The image must be in shape (M, N, 3)")
        if not is_numeric:
            raise NumericTypeError("The image must be numerical")

class NumericTypeError(Exception):
    pass

class IncorrectShapeError(Exception):
    pass
