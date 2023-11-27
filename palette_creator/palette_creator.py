import numpy as np
from tqdm import tqdm
from palette_creator.methods import KMeans, Method, MedianCut, PNNQuantization


class PaletteCreator:
    """API Class for palette creators"""

    def __init__(self, method: str, num_colors: int = 6, optimize_palette: bool = False, **kwargs):
        """

        :param str method: method used for creating the palette. Valid ones: 'kmeans', ...
        :param num_colors: Optional size of the palette. Default 6.
        :returns: None

        """

        self.method = method
        self.optimize_palette = optimize_palette
        self.model = self.__init_method(method, num_colors, **kwargs)
        self.num_colors = num_colors
        if self.optimize_palette:
            self.num_colors += 2

    def create_palette(self, images: list[np.ndarray]) -> tuple[list, list]:
        """Create the palettes of a list of images

        :param list[np.ndarray] images: list of images
        :returns:

        """
        palettes, proportions = [], []
        for i in tqdm(range(len(images))):
            image = images[i]
            self.__validate_image(image)
            try:
                palette_img, proportions_img = self.model.create_palette(image)
                if self.optimize_palette:
                    palette_img, proportions_img = self.get_optimized_palette(palette_img, proportions_img)
            except Exception as err:
                print(f"Error in image {i}")
                raise err
            # Sort the colors and proportions by proportions
            palette_img, proportions_img = zip(
                *sorted(zip(palette_img, proportions_img), key=lambda x: x[1], reverse=True)
            )
            palette_img = np.array(palette_img)
            proportions_img = np.array(proportions_img)
            palettes.append(palette_img)
            proportions.append(proportions_img)
        return palettes, proportions

    def __init_method(self, method, num_colors, **kwargs) -> Method:
        if method == "kmeans":
            return KMeans(n_clusters=num_colors, n_init="auto")
        elif method == "median-cut":
            return MedianCut(palette_colors=num_colors)
        elif method == "pnn":
            return PNNQuantization(palette_colors=num_colors, max_iterations=10000, initial_clusters=350, **kwargs)

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
        
    def get_optimized_palette(self, palette, proportions):
        palette_ = np.array(palette).astype(int)
        proportions = np.array(proportions)
        new_palette = []
        remaining = []
        remaining_props = []
        new_proportions = []

        # Find the nearest neighbor (closest color) for each color in the palette
        distances = np.linalg.norm(palette_[:, np.newaxis, :] - palette_[np.newaxis, :, :], axis=2)
        NN = np.partition(distances, 1, axis=1)[:, 1]
        NN_sorted = np.argsort(NN)[-1::-1]

        i = 0
        r = 0

        # Find the 6 most different colors
        while len(new_palette) < self.num_colors - 2:
            # If there are no more different colors, add the remaining colors to complete the palette
            if i >= len(NN_sorted):
                remaining_slice = slice(r, r + 6 - len(new_palette))
                new_palette.extend(remaining[remaining_slice])
                new_proportions.extend(remaining_props[remaining_slice])
                r += 6 - len(new_palette)
                continue
            
            # When colors have the same distance, choose the one with the biggest proportion
            NN_distance = NN[NN_sorted[i]]
            neighbors_mask = NN == NN_distance

            props = proportions[neighbors_mask]
            biggest = np.max(props)
            biggest_palette_element = palette_[proportions == biggest][0]

            remaining.extend(palette_[np.in1d(proportions, props[props != biggest])])
            remaining_props.extend(props[props != biggest])

            new_palette.append(biggest_palette_element)
            new_proportions.append(biggest)
            i += np.sum(neighbors_mask)

        return new_palette, new_proportions


class NumericTypeError(Exception):
    pass


class IncorrectShapeError(Exception):
    pass
