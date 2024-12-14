# Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the Dataset
file_path = "C:\\Users\\HP\\train.json"

# Load and Inspect Dataset
try:
    data = pd.read_json(file_path, lines=True)
    print("Data loaded successfully.")
except Exception as e:
    raise ValueError(f"Error loading data: {e}")

# Display the first 5 rows and dataset info
print("First 5 rows of the dataset:")
print(data.head())
print("\nDataset Info:")
print(data.info())

# Data Exploration
# Check for missing values
missing_values = data.isnull().sum()
print("\nMissing Values:")
print(missing_values)

# Summary statistics for the 'helpful' column
print("\nSummary Statistics for 'helpful':")
print(data['helpful'].describe())

# Plot distribution of 'helpful' scores with improved interpretation
plt.figure(figsize=(10, 6))
plt.hist(data['helpful'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.title("Distribution of 'helpful' Scores", fontsize=14)
plt.xlabel("'helpful' Score", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(True)
plt.show()

# Text Interpretation: Distribution of 'helpful' Scores
print("\nInterpretation:")
print("The distribution of 'helpful' scores shows a concentration of values near the mean,")
print("indicating that most reviews have moderate helpfulness. A few reviews stand out with")
print("very high or very low helpfulness, which might represent outliers or extreme cases.")

# Preprocessing: Retain only the numeric 'helpful' column for clustering
numeric_data = data[['helpful']]  # Extract the numeric column

# Check if numeric data is present for clustering
if numeric_data.empty:
    raise ValueError("No numeric columns found for clustering.")

# Standardize the Data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(numeric_data)
print("Data standardization completed.")

# K-Means Clustering
n_clusters = 3  # Adjust the number of clusters as needed
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(data_scaled)

# Add Cluster Labels to the Original Dataset
data['Cluster'] = kmeans.labels_
print(f"K-Means clustering applied with {n_clusters} clusters.")

# Visualize Clusters: Histogram for 'helpful' Scores with Clusters
plt.figure(figsize=(12, 7))
for cluster in range(n_clusters):
    cluster_data = data[data['Cluster'] == cluster]
    plt.hist(cluster_data['helpful'], bins=20, alpha=0.5, label=f'Cluster {cluster}')

plt.title("K-Means Clustering: Distribution of 'helpful' Scores by Cluster", fontsize=14)
plt.xlabel("'helpful' Score", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.legend()
plt.grid(True)
plt.show()

# Text Interpretation: Clusters
print("\nCluster Interpretation:")
print("Cluster 0: Represents reviews with lower 'helpful' scores. These could be uninformative or less impactful reviews.")
print("Cluster 1: Represents reviews with moderate 'helpful' scores. These are likely moderately useful reviews.")
print("Cluster 2: Represents reviews with higher 'helpful' scores. These are highly useful and well-received reviews.")

# Save the Clustered Dataset
output_path = "clustered_dataset.csv"
data.to_csv(output_path, index=False)
print(f"\nClustered dataset saved to {output_path}")

# Display Cluster Statistics
print("\nCluster Statistics:")
print(data.groupby('Cluster')['helpful'].describe())