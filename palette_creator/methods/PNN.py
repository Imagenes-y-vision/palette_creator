import numpy as np
import cv2
from palette_creator.methods.abstract_method import Method
import matplotlib.pyplot as plt


class PNN:
    def __init__(self, num_clusters, max_iterations=100, initial_clusters=100, palette_method='mode'):
        self.num_clusters = num_clusters
        self.clusters = []
        self.max_iterations = max_iterations
        self.initial_clusters = initial_clusters
        self.palette_method = palette_method

    def initialize_clusters(self, image):
        # Create 100 uniformly distributed clusters
        # ordered_pixels = sorted(image, key=lambda x: np.mean(x))
        # self.clusters = np.array_split(ordered_pixels, self.initial_clusters)
        max = (image).max(axis=0)
        min = (image).min(axis=0)
        centroids = np.linspace(min, max, self.initial_clusters)
        self.clusters = [[] for _ in centroids]
        for i in range(len(image)):
            self.clusters[np.argmin(np.linalg.norm(image[i] - centroids, axis=1))].append(image[i])
        self.clusters = [np.array(cluster) for cluster in self.clusters if len(cluster) > 0]

    def find_best_combination(self, clusters):
        combs = []
        impact = []

        combinations = PairCombinations(clusters, 10)
        for combo in combinations:
            c1_id, c2_id = combo

            # Original formula of distance between clusters
            cluster_centroids = [np.mean(clusters[c1_id], axis=0), np.mean(clusters[c2_id], axis=0)]
            distance = np.linalg.norm(cluster_centroids[0] - cluster_centroids[1])

            # Find the impact on the mse of merging these two clusters
            # mse_impact = distance ** 2 * (len(clusters[c1_id]) * len(clusters[c2_id])) / (len(clusters[c1_id]) + len(clusters[c2_id])) # Original formula
            mse_impact = distance ** 2 # Do not take into account the number of pixels in the clusters

            combs.append((c1_id, c2_id))
            impact.append(mse_impact)
        
        # Find the combination with the lowest impact
        return combs[np.argmin(impact)]

    def merge_clusters(self, c1_id, c2_id):
        cluster1 = self.clusters[c1_id]
        cluster2 = self.clusters[c2_id]
        new_cluster = np.concatenate((cluster1, cluster2))
        # remove the old clusters
        try:
            del self.clusters[c2_id]
            del self.clusters[c1_id]
        except Exception as e:
            print('Ids to merge', c1_id, c2_id, len(self.clusters))
            raise e
        # add the new cluster
        self.clusters.insert(c1_id, new_cluster)
        # Sort the clusters by their mean
        self.clusters = sorted(self.clusters, key=lambda x: np.mean(x))


    def fit(self, img):
        self.img_size = len(img)
        self.initialize_clusters(img)

        for _ in range(self.max_iterations):
            if len(self.clusters) > self.num_clusters:
                clusters_to_merge = self.find_best_combination(self.clusters)
                self.merge_clusters(*clusters_to_merge)
            else:
                break

    def get_palette(self):
        if self.palette_method == 'mode':
            palette_img = []
            for cluster in self.clusters:
                pixels, counts= np.unique(cluster, axis=0, return_counts=True)
                palette_img.append(pixels[np.argmax(counts)])
        elif self.palette_method == 'mean':
            palette_img = [np.mean(cluster, axis=0) for cluster in self.clusters]
        elif self.palette_method == 'median':
            palette_img = [np.median(cluster, axis=0) for cluster in self.clusters]
        else:
            raise NotImplementedError
        proportions_img = [len(cluster) / self.img_size for cluster in self.clusters]

        return palette_img, proportions_img
    
    def show_mapped_img(self, img):
        palette, _ = self.get_palette()
        palette = np.array(palette)
        img_reshape = img.reshape(-1, 3)
        mapped_img = np.copy(img_reshape).astype(int)
        for i, cluster in enumerate(self.clusters):
            for c_pixel in cluster:
                condition = np.all(mapped_img == c_pixel, axis=1)
                mapped_img[condition] = palette[i]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(img)
        ax2.imshow(mapped_img.reshape(img.shape))
        plt.show()


class PNNQuantization(Method):
    """PNNQuantization palette creator."""

    def __init__(
        self,
        palette_colors=6,
        max_iterations=200,
        initial_clusters=100,
        palette_method='mean'
    ):
        self.palette_colors = palette_colors
        self.max_iterations = max_iterations
        self.initial_clusters = initial_clusters
        self.palette_method = palette_method

    def create_palette(self, image: np.ndarray) -> tuple[list, list]:
        """Train the model and create the palette.

        Args:
            image: image to be processed.

        Returns:
            tuple: palette and proportions.
        """
        # reshape to vector of features, where a feature is the color (a feature is a vector of 3 dimensions, i.e., rgb)
        img_reshape = image.reshape(-1, 3).astype(int)

        # Rezise image if it is too big
        while len(img_reshape) > 10000:
            image = cv2.resize(image, None, fx = 0.75, fy = 0.75)
            img_reshape = image.reshape(-1, 3)

        self.pnn_model = PNN(self.palette_colors, max_iterations=self.max_iterations, initial_clusters=self.initial_clusters, palette_method=self.palette_method)
        self.pnn_model.fit(img_reshape)

        palette, proportions = self.pnn_model.get_palette()
        
        return palette, proportions


class PairCombinations:
    """Create an iterator that returns all possible combinations of pairs of elements from a list within a range of distance."""

    def __init__(self, lst, distance):
        self.lst = list(range(len(lst)))
        self.distance = distance
        self.curr_id = 0
        self.second_id = 1
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.curr_id >= len(self.lst):
            raise StopIteration
        
        if self.second_id >= len(self.lst):
            self.curr_id += 1
            self.second_id = self.curr_id + 1
            return self.__next__()
        
        # Check if the distance between the two elements is within the range
        if self.second_id - self.curr_id <= self.distance:
            # If so, return the pair and increment the second id
            pair = (self.curr_id, self.second_id)
            self.second_id += 1
            return pair
        else:
            # Otherwise, increment the current_id and reset the second_id
            self.curr_id += 1
            self.second_id = self.curr_id + 1
            return self.__next__()