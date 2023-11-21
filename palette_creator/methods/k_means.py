import numpy as np
from sklearn.cluster import KMeans as SKLearnKMeans
from palette_creator.methods.abstract_method import Method


class KMeans(Method, SKLearnKMeans):
    """K-means palette creator."""

    def __init__(
        self,
        n_clusters=8,
        *,
        init="k-means++",
        n_init="warn",
        max_iter=300,
        tol=1e-4,
        verbose=0,
        random_state=None,
        copy_x=True,
        algorithm="lloyd",
    ):
        super().__init__(
            n_clusters,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            random_state=random_state,
            copy_x=copy_x,
            algorithm=algorithm,
        )

    def create_palette(self, image: np.ndarray) -> tuple[list, list]:
        # reshape to vector of features, where a feature is the color (a feature is a vector of 3 dimensions, i.e., rgb)
        img_reshape = image.reshape(-1, 3)

        # the algorithm of clustering is done here via scikit-learn
        self.fit(img_reshape)

        # in Kmeans, the colors of the palette are the centroids
        palette = self.cluster_centers_.astype(int)
        proportions = np.unique(self.labels_, return_counts=True)[1] / len(self.labels_)
        return palette, proportions
