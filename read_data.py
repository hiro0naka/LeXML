import numpy as np
import scipy.sparse as sp
from scipy.io import mmwrite, mmread
import multiprocessing


def get_y(y, data, ind):
    for y_ind in data:
        y[ind, int(y_ind)] = 1

def get_x(x, data, ind):
    for x_data in data:
        x_ind_val = x_data.split(":")
        x[ind, int(x_ind_val[0])] = float(x_ind_val[1])

def get_data(file_name):
    print("FILE: " + file_name)
    f = open(file_name)

    dataset_size = [int(n) for n in f.readline().split(" ")]
    x = np.zeros((dataset_size[0], dataset_size[1]), dtype=np.float32)
    y = np.zeros((dataset_size[0], dataset_size[2]), dtype=np.float32)

    for ind in range(dataset_size[0]):
        xy_data = f.readline().split(" ")
        get_y(y, xy_data[0].split(","), ind)
        get_x(x, xy_data[1:], ind)

    f.close()
    return x, y

def get_spdata(file_name):
    print("FILE: " + file_name)
    f = open(file_name)

    dataset_size = [int(n) for n in f.readline().split(" ")]
    x = sp.lil_matrix((dataset_size[0], dataset_size[1]), dtype=np.float32)
    y = sp.lil_matrix((dataset_size[0], dataset_size[2]), dtype=np.float32)

    for ind in range(dataset_size[0]):
        xy_data = f.readline().split(" ")
        get_y(y, xy_data[0].split(","), ind)
        get_x(x, xy_data[1:], ind)

    f.close()
    return x.tocsr(), y.tocsr()


def get_rep_w(file_name, embedding_method, num_label):
    print("FILE: " + file_name)
    f = open(file_name)

    rep_size = [int(n) for n in f.readline().split(" ")]
    w = np.zeros((num_label, rep_size[1]))

    if embedding_method == "deepwalk" or embedding_method == "node2vec":
        for i in range(rep_size[0]):
            data = f.readline().split(" ")
            w[int(data[0])] = data[1:]

    elif embedding_method == "line":
        for i in range(rep_size[0]):
            data = f.readline().split(" ")
            w[int(data[0])] = data[1:-1]

    f.close()
    return w

def write_spmtx(file_name):
    x, y = get_spdata(file_name)

    file = str(file_name.split(".txt")[0])
    print("FILE: " + file)
    mmwrite(file + "_x", x)
    mmwrite(file + "_y", y)

def read_spmtx(file_name):
    print("FILE: " + file_name)
    x = mmread(file_name + "_x.mtx")
    y = mmread(file_name + "_y.mtx")
    return x.tocsr(), y.tocsr()

# class mp_get_data:
#     def __init__(self, file_name, processes=8):
#         print("FILE: " + file_name)
#         f = open(file_name)
#         self.lines = f.readlines()
#         dataset_size = [int(n) for n in self.lines[0].split(" ")]
#         f.close()
#
#         self.x = sp.lil_matrix((dataset_size[0], dataset_size[1]))
#         self.y = sp.lil_matrix((dataset_size[0], dataset_size[2]))
#
#         process_list = range(dataset_size[0])
#         p = multiprocessing.Pool(processes=processes)
#         p.map(self.get_data, process_list)
#         print(self.x)
#
#     def get_y(self, data, ind):
#         for y_ind in data:
#             self.y[ind, int(y_ind)] = 1
#
#     def get_x(self, data, ind):
#         for x_data in data:
#             x_ind_val = x_data.split(":")
#             self.x[ind, int(x_ind_val[0])] = float(x_ind_val[1])
#
#     def get_data(self, ind):
#         xy_data = self.lines[ind+1].split(" ")
#         self.get_y(xy_data[0].split(","), ind)
#         self.get_x(xy_data[1:], ind)
#
#     def return_data(self):
#         return self.x.tocsr(), self.y.tocsr()


# debug
if __name__ == "__main__":
    # x, y = get_data("dataset/Wiki10/wiki10_train.txt")

    write_spmtx("dataset/Amazon/amazon_train.txt")
    write_spmtx("dataset/Amazon/amazon_test.txt")

    # write_spmtx("dataset/Wiki10/wiki10_train.txt")
    # x, y = read_spmtx("dataset/Wiki10/wiki10_train")