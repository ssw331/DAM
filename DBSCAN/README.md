# README

## Imported Module

Here is the modules imported.

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import DistanceMetric, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
```

## Core Implementation of DBSCAN

The is_core() is one of the core function of DBSCAN. In each process of is_core(), it will search if there are some unprocessed(marks[i] < 1) directly-reachable points. 

If there are, then collect this group of points and mark these point as processed(marks[i] = 1). Then, it will traverse all the points in the group to calculate other reachable points by DFS in the group. For each point , is_core() will be called to calculate the directly-reachable points from the points in the group, so that we can get the reachable points. Finally, it will check the amount of points in the group of reachable points. If it is bigger than min_smaples, it will return a true to mark it is a cluster; or it will return a false to mark it is a outlier.

If there are not, then return the function with the unchanged cluster and false which means there is no directly-reachable points.

```python
def is_core(idx, dist, marks, eps, min_samples):
    cluster = [idx]

    for each in range(len(dist[0])):
        if dist[idx][each] <= eps and each != idx and marks[each] < 1:  # 查找是否存在尚未处理过的从idx出发的可直接到达点
            cluster.append(each)
            marks[each] = 1

    if cluster == [idx]:  # 意味着这个idx已经没有其他可直接到达点未被处理了
        return False, cluster

    for e in cluster:  # 对新增的可到达点进行深度搜索
        if e != idx:
            _, new_cluster = is_core(e, dist, marks, eps, min_samples)  # 收集新的可到达点
            cluster = list(set(cluster).union(set(new_cluster)))  # 合并旧的和新的

    if len(cluster) >= min_samples:
        return True, cluster
    else:
        return False, cluster
```

The dbscan() is used to prepare the matrix we need such as marks which is used to mark the processed points, dist which contains all the distances from one point to another one. Then it will traverse all the points to check if they are the core points and classify the group gotten into outlier or cluster.

```python
def dbscan(x, eps, min_samples):
    clusters = []
    outlier = []
    marks = np.zeros(len(x))
    dist = DistanceMetric.get_metric("euclidean")
    dist = dist.pairwise(x)
    for seq in range(len(x)):
        if marks[seq] < 1:
            marks[seq] = 1
            judge, new_cluster = is_core(seq, dist, marks, eps, min_samples)
            if judge:
                clusters.append(new_cluster)
            else:
                for r in range(len(new_cluster)):
                    outlier.append(new_cluster[r])

    return clusters, outlier
```

## Data Pre-Processing

At this part, data will be loaded and pre-processed for next process of DBSCAN. First, normalization will be conducted to reduce the effect of some attributes with large scale. Then PCA will reduce the dimensions of data to 2 for easily-processing since 13 attributes can not be easily clustered. By the way, dataset is from [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records).

```python
data = pd.read_csv("../dataset/heart+failure+clinical+records/heart_failure_clinical_records_dataset.csv")

scaler = StandardScaler()
normalized_data = scaler.fit_transform(data)
normalized_data = pd.DataFrame(normalized_data, index=data.index, columns=data.columns)

pca = PCA(n_components=2)
featured_data = pca.fit_transform(normalized_data)
```

## DBSCAN

Then call the dbscan() to use DBSCAN to cluster the data. I have tried seven group of parameters for tuning. And further details can be read in the report. After that, labels will be created to calculate the Davies Bouldin Score of the clustering. In the labels, the outliers is marked as -1, and other clusters will be marked as 0, 1, 2,.... One same number means the same cluster.

```python
# res_clusters, res_outliers = dbscan(featured_data, 0.2, 4)
# res_clusters, res_outliers = dbscan(featured_data, 0.3, 4)
# res_clusters, res_outliers = dbscan(featured_data, 0.4, 4)
# res_clusters, res_outliers = dbscan(featured_data, 0.5, 4)
# res_clusters, res_outliers = dbscan(featured_data, 0.2, 11)
res_clusters, res_outliers = dbscan(featured_data, 0.2, 6)
# res_clusters, res_outliers = dbscan(featured_data, 0.1, 4)

labels = np.zeros(len(featured_data), dtype=int)

for i in range(len(res_clusters)):
    for j in range(len(res_clusters[i])):
        labels[res_clusters[i][j]] = i

for i in range(len(res_outliers)):
    labels[res_outliers[i]] = -1
```

## Result Display

Then print all the results, including clusters, outliers and Davies Bouldin Score of this clustering.

```python
print("-----------------------------------------")

for i in range(len(res_clusters)):
    print((res_clusters[i]))  # 30

print("-----------------------------------------")

print(len(res_outliers))

print("-----------------------------------------")

print("davies_bouldin_score: ", davies_bouldin_score(featured_data, labels))
```

Finally, draw the plot of clustering result.

```python
colors = ['red', 'blue', 'green', 'brown', 'cyan']
markers = ['o', 'v', '^', '<', 'x', '>']

plt.figure(figsize=(10, 10))
plt.xlabel("pca0")
plt.ylabel("pca1")
plt.title("DBSCAN")

for i in range(len(res_clusters)):
    for j in res_clusters[i]:
        plt.scatter(featured_data[j, :1], featured_data[j, 1:], color=colors[i % len(colors)],
                    marker=markers[i % len(markers)])

for i in range(len(res_outliers)):
    plt.scatter(featured_data[res_outliers[i], :1], featured_data[res_outliers[i], 1:],
                color=colors[len(res_clusters) % len(colors)], marker=markers[len(res_clusters) % len(markers)])

plt.show()
```

