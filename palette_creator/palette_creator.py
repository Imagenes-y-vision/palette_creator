import numpy as np
from palette_creator.methods import KMeans, Model


class PaletteCreator:
    """API Class for palette creators"""

    def __init__(self, method: str, num_colors: int = 6):
        """

        :param str method: method used for creating the palette. Valid ones: 'kmeans', ...
        :param num_colors: Optional size of the palette. Default 6.
        :returns: None

        """

        self.method = method
        self.__model = self.__init_model(method, num_colors)
        self.num_colors = num_colors

    def create_palette(self, images: list[np.ndarray]) -> list[tuple[list, list]]:
        """Create the palettes of a list of images

        :param list[np.ndarray] images: list of images
        :returns:

        """
        results = []
        for image in images:
            self.__validate_image(image)
            palette, proportions = self.__model.create_palette(image)
            # Sort the colors and proportions by proportions
            palette, proportions = zip(
                *sorted(zip(palette, proportions), key=lambda x: x[1], reverse=True)
            )
            results.append((palette, proportions))
        return results

    def __init_model(self, method, num_colors) -> Model:
        if method == "kmeans":
            return KMeans(n_clusters=num_colors, n_init="auto")

        else:
            raise NotImplementedError

    @staticmethod
    def __validate_image(image: np.ndarray) -> None:
        """Validate if the image has the correct shape (M, N, 3)

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
