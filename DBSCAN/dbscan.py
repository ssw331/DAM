import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import DistanceMetric, davies_bouldin_score
from sklearn.preprocessing import StandardScaler


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


data = pd.read_csv("../dataset/heart+failure+clinical+records/heart_failure_clinical_records_dataset.csv")

scaler = StandardScaler()
normalized_data = scaler.fit_transform(data)
normalized_data = pd.DataFrame(normalized_data, index=data.index, columns=data.columns)

pca = PCA(n_components=2)
featured_data = pca.fit_transform(normalized_data)

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

print("-----------------------------------------")

for i in range(len(res_clusters)):
    print((res_clusters[i]))  # 30

print("-----------------------------------------")

print(len(res_outliers))

print("-----------------------------------------")

print("davies_bouldin_score: ", davies_bouldin_score(featured_data, labels))

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
