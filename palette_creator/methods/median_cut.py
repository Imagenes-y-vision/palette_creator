import numpy as np
import cv2
from palette_creator.methods.abstract_method import Method


class MedianCut(Method):
    """MedianCut palette creator."""

    def __init__(
        self,
        palette_colors=6,
    ):
        """MedianCut palette creator.

        Args:
            palette_colors: number of colors of the palette.
        """
        self.palette_colors = palette_colors

    def create_palette(self, image: np.ndarray) -> tuple[list, list]:
        """Median cut algorithm.

        Args:
            image: image to be processed.

        Returns:
            tuple: palette and proportions.
        """
        # reshape to vector of features, where a feature is the color (a feature is a vector of 3 dimensions, i.e., rgb)
        img_reshape = image.reshape(-1, 3).astype(int)

        # Rezise image if it is too big
        while len(img_reshape) > 1000000:
            image = cv2.resize(image, None, fx = 0.75, fy = 0.75)
            img_reshape = image.reshape(-1, 3)

        # get the initial box
        box = [img_reshape]

        # split the box until the number of boxes is equal to the number of clusters
        while len(box) < self.palette_colors:
            box_to_split = -1 # get the group with the largest range
            box1 = []
            box2 = []
            
            # get the box with the largest side
            greatest_range_for_box = np.argsort(np.max([np.max(box[i], axis=0) - np.min(box[i], axis=0) for i in range(len(box))], axis=1))
            largest_box = greatest_range_for_box[box_to_split]
            # get the largest side of the box
            range_by_channel = np.argsort(np.max(box[largest_box], axis=0) - np.min(box[largest_box], axis=0))
            largest_side = range_by_channel[-1]

            # get the median of the largest side
            median = np.median(box[largest_box][:, largest_side])
            # split the box in two
            box1 = box[largest_box][box[largest_box][:, largest_side] <= median]
            box2 = box[largest_box][box[largest_box][:, largest_side] > median]

            # if the split was not successful, try again with the second largest side
            side = -2
            while len(box1) == 0 or len(box2) == 0:
                largest_side = range_by_channel[side]
                median = np.median(box[largest_box][:, largest_side])
                box1 = box[largest_box][box[largest_box][:, largest_side] <= median]
                box2 = box[largest_box][box[largest_box][:, largest_side] > median]
                side -= 1

                if side == -4:
                    break

            # if the split was not successful, divide the largest box in two
            if len(box1) == 0 or len(box2) == 0:
                box1 = box[largest_box][:len(box[largest_box]) // 2]
                box2 = box[largest_box][len(box[largest_box]) // 2:]
            
            # remove the box that was split
            del box[largest_box]
            # add the two new boxes
            box += [box1, box2]

        # the colors of the palette are the centroids
        palette = [np.mean(box[i], axis=0) for i in range(len(box))]
        proportions = [len(box[i]) / len(img_reshape) for i in range(len(box))]
        return palette, proportions
