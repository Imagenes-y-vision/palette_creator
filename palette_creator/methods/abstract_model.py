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
