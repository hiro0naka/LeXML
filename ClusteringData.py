import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier


def clustering(data, num_partition):
    cls = KMeans(num_partition, n_jobs=20)
    c_predict = cls.fit.predict(data)
    centers = cls.cluster_centers_
    return cls, c_predict, centers

def cls_partition(data_x, data_y, cls, c_predict, centers, num_partition, n_neighbors, sparse):
    Z = []
    for i in range(num_partition):
        c_list = [j for j, xc in enumerate(c_predict) if xc == i]
        Z.append(partition_data(data_x[c_list], data_y[c_list], centers[i], n_neighbors, sparse))
    return Z


class partition_data():
    def __init__(self, data_x, data_y, center, n_neighbors, sparse):
        self.sparse = sparse
        self.center = center
        self.num_data = data_x.shape[0]
        print("num_data: " + str(self.num_data))
        self.data_x = data_x
        self.data_y = data_y
        self.n_neighbors = n_neighbors
        self.setup_knn()

    def setup_knn(self):
        self.cls = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        self.cls.fit(self.data_x, self.data_y)

    def return_center(self):
        return self.center

    def return_knn(self, q):
        return self.cls.kneighbors(q, return_distance=False)

    def return_predict(self, q, k, k_pred=5):
        max_k = range(0, k_pred)
        sum_y = np.sum(self.data_y[self.return_knn(q)[0,0:k]], axis=0)
        if self.sparse:
            predict = np.argsort(-sum_y)[0, max_k]
        else:
            predict = np.argsort(-sum_y)[max_k]
        return predict