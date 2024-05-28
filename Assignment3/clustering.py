import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import fastcluster
from scipy.cluster.hierarchy import dendrogram


df = pd.read_csv('diabetes_prediction_dataset.csv')


df = df[df['smoking_history'] != 'No Info']


label_encoder = LabelEncoder()
df['gender'] = label_encoder.fit_transform(df['gender'])
df['smoking_history'] = label_encoder.fit_transform(df['smoking_history'])


features = ['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']


scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[features])

#Elbow Method
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_scaled)
    sse.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), sse, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.title('Elbow Method')
plt.show()

# K-Means
kmeans = KMeans(n_clusters=6, random_state=42)  # Adjust the number of clusters as needed
kmeans.fit(df_scaled)

df['cluster'] = kmeans.labels_

#size of each cluster
cluster_sizes = df['cluster'].value_counts()
print("Cluster Sizes:\n", cluster_sizes)

# cluster centroids
centroids = kmeans.cluster_centers_
centroids_df = pd.DataFrame(scaler.inverse_transform(centroids), columns=features)
print("Cluster Centroids:\n", centroids_df)

# Visualize the clusters using PCA
pca = PCA(n_components=2)
pca_components = pca.fit_transform(df_scaled)

df['pca1'] = pca_components[:, 0]
df['pca2'] = pca_components[:, 1]

# Calculate the PCA coordinates of the centroids
centroids_pca = pca.transform(centroids)

plt.figure(figsize=(10, 6))
sns.scatterplot(x='pca1', y='pca2', hue='cluster', data=df, palette='viridis')
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], s=300, c='red', marker='X', edgecolor='black', linewidth=2, label='Centroids')
plt.title('Cluster Visualization with Centroids')
plt.legend()
plt.show()


sample_df = df.sample(n=1000, random_state=42)  # Adjust the sample size as needed
sample_df_scaled = scaler.transform(sample_df[features])

# Hierarchical Clustering using Single Linkage with fastcluster
linked_single = fastcluster.linkage(sample_df_scaled, method='single')

plt.figure(figsize=(10, 6))
dendrogram(linked_single, truncate_mode='level', p=5)
plt.title('Hierarchical Clustering Dendrogram (Single Linkage)')
plt.xlabel('Sample index')
plt.ylabel('Distance')
plt.show()

# Hierarchical Clustering using Complete Linkage with fastcluster
linked_complete = fastcluster.linkage(sample_df_scaled, method='complete')

plt.figure(figsize=(10, 6))
dendrogram(linked_complete, truncate_mode='level', p=5)
plt.title('Hierarchical Clustering Dendrogram (Complete Linkage)')
plt.xlabel('Sample index')
plt.ylabel('Distance')
plt.show()

# Apply DBSCAN for Density-Based Clustering with adjusted parameters
dbscan = DBSCAN(eps=0.3, min_samples=10)  # Adjust eps and min_samples as needed
dbscan_labels = dbscan.fit_predict(df_scaled)


df['dbscan_cluster'] = dbscan_labels

# size of each cluster
dbscan_cluster_sizes = df['dbscan_cluster'].value_counts()
print("DBSCAN Cluster Sizes:\n", dbscan_cluster_sizes)

# noise points
df_filtered = df[df['dbscan_cluster'] != -1]

# Visualize the DBSCAN clusters using PCA
plt.figure(figsize=(10, 6))
sns.scatterplot(x='pca1', y='pca2', hue='dbscan_cluster', data=df_filtered, palette='viridis', legend='full')
plt.title('DBSCAN Cluster Visualization')
plt.show()