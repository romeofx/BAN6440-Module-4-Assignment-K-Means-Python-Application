import unittest
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class TestKMeansClustering(unittest.TestCase):

    def setUp(self):
        # Sample data to test clustering
        self.data = pd.DataFrame({
            "Feature1": [1, 2, 3, 4, 5],
            "Feature2": [5, 4, 3, 2, 1]
        })
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=2, random_state=42)

    def test_data_standardization(self):
        """Test the data standardization process."""
        # Standardize the data and check if the shape is correct
        scaled_data = self.scaler.fit_transform(self.data)
        self.assertEqual(scaled_data.shape, self.data.shape)
        self.assertTrue(np.allclose(np.mean(scaled_data, axis=0), 0), "Mean of the scaled data is not close to 0")
        self.assertTrue(np.allclose(np.std(scaled_data, axis=0), 1), "Standard deviation of the scaled data is not 1")

    def test_kmeans_clustering(self):
        """Test the KMeans clustering algorithm."""
        # Standardize the data
        scaled_data = self.scaler.fit_transform(self.data)

        # Fit the KMeans model
        self.kmeans.fit(scaled_data)

        # Check if the number of clusters matches the expected number
        self.assertEqual(len(np.unique(self.kmeans.labels_)), self.kmeans.n_clusters,
                         "Number of clusters doesn't match")

        # Test if the KMeans labels are assigned to the correct number of samples
        self.assertEqual(len(self.kmeans.labels_), len(self.data),
                         "Number of labels doesn't match the number of data points")

        # Optionally, check if some of the cluster centers are distinct (which is expected)
        cluster_centers = self.kmeans.cluster_centers_
        self.assertGreater(np.linalg.norm(cluster_centers[0] - cluster_centers[1]), 0,
                           "Cluster centers are too close to each other")

    def test_cluster_assignments(self):
        """Test whether data points are correctly assigned to clusters."""
        scaled_data = self.scaler.fit_transform(self.data)
        self.kmeans.fit(scaled_data)

        # Cluster assignments should match the number of samples
        cluster_assignments = self.kmeans.labels_
        self.assertEqual(len(cluster_assignments), len(self.data), "Cluster assignment count mismatch")

        # Ensure each cluster has at least one sample
        unique, counts = np.unique(cluster_assignments, return_counts=True)
        for cluster_id, count in zip(unique, counts):
            self.assertGreater(count, 0, f"Cluster {cluster_id} has no data points")

    def test_cluster_centers(self):
        """Test if the cluster centers are calculated properly."""
        scaled_data = self.scaler.fit_transform(self.data)
        self.kmeans.fit(scaled_data)

        # Check if the cluster centers are not NaN and they are distinct
        cluster_centers = self.kmeans.cluster_centers_
        self.assertFalse(np.any(np.isnan(cluster_centers)), "Cluster centers contain NaN values")
        self.assertTrue(np.all(cluster_centers != 0), "Cluster centers should not be all zero")

if __name__ == "__main__":
    unittest.main(argv=[''], exit=False)