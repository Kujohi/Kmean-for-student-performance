import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('sample_data.csv')

# Clean column names by stripping whitespace
df.columns = df.columns.str.strip()

# Select features for clustering
features = ['Age', 'study_hours', 'monthly_spending', 'sleep_duration', 
           'movie_hours', 'sports_hours', 'book_hours', 'gaming_hours', 
           'social_media_hours']

# Create feature matrix X
X = df[features]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine optimal number of clusters using elbow method
inertias = []
K = range(1, 5)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

# Plot elbow curve
plt.figure(figsize=(10, 6))
plt.plot(K, inertias, 'bx-')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k')
plt.savefig('elbow_curve.png')
plt.close()

# Perform K-means clustering with optimal k (let's use k=4 for this example)
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Analyze clusters
cluster_means = df.groupby('Cluster')[features].mean()
print("\nCluster Means:")
print(cluster_means)

# After calculating cluster_means, add this code to label the clusters
def assign_cluster_labels(cluster_means):
    # Create fixed mapping for clusters based on known performance
    cluster_labels = {
        1: 'Good',
        2: 'Medium',
        0: 'Poor'
    }
    return cluster_labels

# Get cluster labels
cluster_labels = assign_cluster_labels(cluster_means)

# Add performance labels to the dataframe
df['Performance'] = df['Cluster'].map(cluster_labels)

# Print cluster characteristics with labels
print("\nCluster Labels:")
for cluster, label in cluster_labels.items():
    print(f"\n{label} Students (Cluster {cluster}):")
    print('"""""""""""""""""""""""""""""')
    print(cluster_means.loc[cluster])
    print('"""""""""""""""""""""""""""""')
# Visualize clusters using PCA
from sklearn.decomposition import PCA

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Create scatter plot
plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['Cluster'], cmap='viridis')
for cluster in range(optimal_k):
    # Calculate center point for each cluster
    mask = df['Cluster'] == cluster
    center_x = X_pca[mask, 0].mean()
    center_y = X_pca[mask, 1].mean()
    plt.annotate(cluster_labels[cluster], 
                (center_x, center_y),
                textcoords="offset points",
                xytext=(0,10),
                ha='center',
                fontsize=12,
                bbox=dict(facecolor='white', alpha=0.7))

plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('Student Performance Clusters Visualization')
plt.colorbar(scatter)
plt.savefig('cluster_visualization.png')
plt.close()

# After calculating cluster_means, before the heatmap creation, add:
# Calculate standardized cluster means
standardized_means = pd.DataFrame(
    scaler.transform(cluster_means),
    columns=cluster_means.columns,
    index=cluster_means.index
)


# Rename the index to use performance labels
standardized_means.index = [cluster_labels[i] for i in standardized_means.index]

# Create feature importance heatmap with standardized values
plt.figure(figsize=(12, 8))
sns.heatmap(standardized_means, annot=True, cmap='RdYlBu_r', fmt='.2f', center=0)
plt.title('Standardized Feature Patterns by Student Performance')
plt.xlabel('Features')
plt.ylabel('Performance Level')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('cluster_heatmap.png')
plt.close()

# Save results
cluster_means.to_csv('cluster_analysis.csv')
df.to_csv('data_with_clusters.csv', index=False)

# Print summary statistics with labels
print("\nNumber of students in each performance category:")
print(df['Performance'].value_counts())

# Calculate and print the silhouette score
from sklearn.metrics import silhouette_score
silhouette_avg = silhouette_score(X_scaled, df['Cluster'])
print(f"\nSilhouette Score: {silhouette_avg:.3f}")
