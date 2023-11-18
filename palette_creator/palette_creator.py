import numpy as np
from palette_creator.methods import KMeans, Model


class PaletteCreator:
    """Base class for palette creators."""

    def __init__(self, method: str, num_colors: int = 6):
        """

        :param str method: method used for cerating the palette. Valid ones: 'kmeans', ...
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
