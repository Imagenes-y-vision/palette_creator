import numpy as np

class Base_palette_creator:
    """Base class for palette creators."""
    
    def __init__(self, palette_name, num_colors = 6):
        """Initialize palette creator.

        Args:
            palette_name (str): Name of palette creator.
            num_colors (int, optional): Number of colors in palette. Defaults to 6.
        """
        self.palette_name = palette_name
        self.num_colors = num_colors

    def create_palette(self, images: list[np.ndarray]) -> list[tuple]:
        """Create palette from images.

        Args:
            images (list): List of images.

        Returns:
            list[tuple]: List of  a tupple with the palette colors the proportion of pixels in the image for each color.
        """

        raise NotImplementedError

    def get_palette_name(self):
        """Get palette name."""
        return self.palette_name
    
    def get_num_colors(self):
        """Get number of colors in palette."""
        return self.num_colors