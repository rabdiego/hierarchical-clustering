# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

# Importing data
data = pd.read_csv('Mall_Customers.csv')
X = data.iloc[:, [3, 4]].values

# Preparing the plot
fig1, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(16, 8))

# Plotting the dendrogram
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
ax2.set_title('Dendrogram', fontsize=26)
ax2.set_xlabel('Customers', fontsize=20)
ax2.set_ylabel('Euclidian Distance', fontsize=20)

# Predicting results
hc = AgglomerativeClustering(n_clusters=5, metric='euclidean', linkage='ward')
y_hc = hc.fit_predict(X)

# Plotting data
ax1.set_title('Clusters', fontsize=26)
ax1.set_xlabel('Anual Income', fontsize=20)
ax1.set_ylabel('Spending Score', fontsize=20)
ax1.scatter(X[y_hc==0, 0], X[y_hc==0, 1], c='r', label='Cluster 1')
ax1.scatter(X[y_hc==1, 0], X[y_hc==1, 1], c='b', label='Cluster 2')
ax1.scatter(X[y_hc==2, 0], X[y_hc==2, 1], c='g', label='Cluster 3')
ax1.scatter(X[y_hc==3, 0], X[y_hc==3, 1], c='y', label='Cluster 4')
ax1.scatter(X[y_hc==4, 0], X[y_hc==4, 1], c='m', label='Cluster 5')
ax1.legend(loc=7)
fig1.savefig('plot.png')
