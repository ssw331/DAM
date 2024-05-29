import heapq
import time

import pandas as pd
import numpy as np
import queue
from sklearn import preprocessing

import matplotlib.pyplot as plt
from scipy.cluster import hierarchy

original_data = pd.read_csv("../dataset/container+crane+controller+data+set/Container_Crane_Controller_Data_Set.csv",
                            sep=';').drop(columns=['Power'])
data = original_data.values
data = preprocessing.minmax_scale(data, axis=0)

# drawing
# Z = hierarchy.linkage(data, method='single')
# plt.figure(figsize=(10, 10))
# hierarchy.dendrogram(Z, labels=np.arange(len(data)))
# plt.show()

# distance pre-processing
# S:n
indices = list(range(len(data)))
# S:n^2
dists = np.zeros((len(indices), len(indices)))

result = []


class PriorityQueue:
    def __init__(self):
        self.heap = queue.PriorityQueue()
        self.time_mark = 0

    def push(self, item, priority):
        self.time_mark = self.time_mark - 1
        self.heap.put((priority, self.time_mark, item))

    def pop(self):
        return self.heap.get()

    def get_min_element(self):
        item = self.heap.get()
        self.heap.put(item)
        return item


# n
def cal_dist(m_x, m_y):
    if type(m_x) is int and type(m_y) is int:
        return np.linalg.norm(data[m_x, :] - data[m_y, :])
    elif type(m_x) is int and type(m_y) is list:
        return min(np.linalg.norm(data[m_x, :] - data[each, :]) for each in m_y)
    elif type(m_x) is list and type(m_y) is int:
        return min(np.linalg.norm(data[each, :] - data[m_y, :]) for each in m_x)
    elif type(m_x) is list and type(m_y) is list:
        return min(np.linalg.norm(data[each, :] - data[j, :]) for each in m_x for j in m_y)


def union(m_x, m_y):
    if type(m_x) is int and type(m_y) is int:
        return [m_x] + [m_y]
    elif type(m_x) is int and type(m_y) is list:
        return [m_x] + m_y
    elif type(m_x) is list and type(m_y) is int:
        return m_x + [m_y]
    elif type(m_x) is list and type(m_y) is list:
        return m_x + m_y
    else:
        return []


# n
def heap_init(dist_matrix):
    heap_matrix = PriorityQueue()
    for row in range(len(dist_matrix)):
        for col in range(len(dist_matrix)):
            heap_matrix.push([row, col], priority=dist_matrix[row][col])

    return heap_matrix


# 1 ~ n * log n
def get_min(hp):
    item = hp.pop()
    i = 0
    while i < len(indices):
        if type(indices[i]) is list:
            if type(item[2][0]) is int and type(item[2][1]) is list:
                if (item[2][0] in indices[i] or ([False for a in item[2][1] if a in indices[i]]
                                                 and set(item[2][1]) != set(indices[i]))):
                    item = hp.pop()
                    i = 0
                    continue
            elif type(item[2][0]) is list and type(item[2][1]) is int:
                if (item[2][1] in indices[i] or ([False for a in item[2][0] if a in indices[i]]
                                                 and set(item[2][0]) != set(indices[i]))):
                    item = hp.pop()
                    i = 0
                    continue
            elif type(item[2][0]) is int and type(item[2][1]) is int:
                if item[2][0] in indices[i] or item[2][1] in indices[i]:
                    item = hp.pop()
                    i = 0
                    continue
            elif type(item[2][0]) is list and type(item[2][1]) is list:
                if ([False for a in item[2][0] if a in indices[i]] and set(item[2][0]) != set(indices[i])) or (
                        [False for a in item[2][1] if a in indices[i]] and set(item[2][1]) != set(indices[i])):
                    item = hp.pop()
                    i = 0
                    continue
        i = i + 1
    return item


# T:n^2
for x in range(len(indices)):
    for y in range(len(indices)):
        if x >= y:
            dists[x, y] = float('inf')
        else:
            dists[x, y] = cal_dist(indices[x], indices[y])

heap = heap_init(dists.tolist())

# T: n ^ 3 ~ n ^ 2 * log n
while len(indices) > 1:
    # T:n * log n
    min_idx = get_min(heap)
    i_x = min_idx[2][0]
    i_y = min_idx[2][1]

    # T:1
    unions = union(i_x, i_y)
    for e in [i_x] + [i_y]:
        indices.remove(e)
    indices.append(unions)

    # T:n * (log n + n)
    for e in range(len(indices) - 1):
        heap.push([indices[-1], indices[e]], cal_dist(indices[e], indices[-1]))

    result.append(unions)

    print('簇: ', str(unions))
    print(indices)
    print('--------------------------------------------------------------------------')
