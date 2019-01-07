import numpy as np
import scipy.sparse as sp
import read_data as rd


def make_adjlist(file_name):
    print("make adjlist: start")
    f = open(file_name)
    dataset_size = [int(n) for n in f.readline().split(" ")]
    num_train = dataset_size[0]
    num_label = dataset_size[2]
    adjlist = np.zeros((num_label, num_label), dtype=np.int32)
    sum_y = np.zeros(num_label, dtype=np.int32)

    for i in range(num_train):
        label_list = f.readline().split(" ")[0].split(",")
        for i, ind1 in enumerate(label_list):
            sum_y[int(ind1)] += 1
            for ind2 in label_list[i+1:]:
                adjlist[int(ind1), int(ind2)] += 1
                adjlist[int(ind2), int(ind1)] += 1

    print("make adjlist: end")
    f.close()
    return adjlist, sum_y

# knn graph
def adjlist_k(embedding_method, k, train_file_name, output_file_name=""):
    adjlist, sum_y = make_adjlist(train_file_name)
    num_label = adjlist.shape[0]
    max_k = range(0, k)

    if embedding_method == "line" or embedding_method == "node2vec":
        f = open(output_file_name + "_k" + str(k) + ".tsv", "a")
        for ind1 in range(num_label):
            print(ind1)
            indices = np.argsort(-adjlist[ind1])[max_k]
            for ind2 in indices:
                if adjlist[ind1, ind2] > 0:
                    f.write(str(ind1) + "\t" + str(ind2) + "\n")
        f.close()

    elif embedding_method == "deepwalk":
        f = open(output_file_name + "_k" + str(k) + ".adjlist", "a")
        for ind1 in range(num_label):
            f.write(str(ind1))
            indices = np.argsort(-adjlist[ind1])[max_k]
            for ind2 in indices:
                if adjlist[ind1, ind2] > 0:
                    f.write(" " + str(ind2))
            f.write("\n")
        f.close()

# weighted graph
def adjlist_w(embedding_method, k, weight_min, train_file_name, output_file_name=""):
    adjlist, sum_y = make_adjlist(train_file_name)
    num_label = adjlist.shape[0]
    max_k = range(0, k)

    if embedding_method == "line" or embedding_method == "node2vec":
        f = open(output_file_name + "_w" + str(k) + ".tsv", "a")
        for ind1 in range(num_label):
            print(ind1)
            if sum_y[ind1] > 0:
                for ind2 in range(num_label):
                    if adjlist[ind1, ind2] > 0:
                        weight = round(adjlist[ind1, ind2] / sum_y[ind1], 3)
                        if weight >= weight_min:
                            f.write(str(ind1) + "\t" + str(ind2) + "\t" + str(weight) + "\n")
        f.close()

# PMI knn graph
def adjlist_p(embedding_method, k, pmi_min, train_file_name, output_file_name=""):
    adjlist, sum_y = make_adjlist(train_file_name)
    num_label = adjlist.shape[0]
    max_k = range(0, k)

    if embedding_method == "line" or embedding_method == "node2vec":
        f = open(output_file_name + "_p" + str(k) + ".tsv", "a")
        for ind1 in range(num_label):
            print(ind1)
            if sum_y[ind1] > 1:
                edge_list = [[round(adjlist[ind1, ind2]/(sum_y[ind1]*sum_y[ind2]), 4), ind2] for ind2 in range(num_label)
                             if adjlist[ind1, ind2] > 0 and adjlist[ind1, ind2]/(sum_y[ind1]*sum_y[ind2]) >= pmi_min]
                edge_list.sort()
                num_edge = len(edge_list) if len(edge_list) < k else k
                for i in range(num_edge):
                    f.write(str(ind1) + "\t" + str(edge_list[-i+1][1]) + "\n")
        f.close()


# large scale
def make_sp_adjlist(file_name):
    print("make adjlist: start")
    f = open(file_name)
    dataset_size = [int(n) for n in f.readline().split(" ")]
    num_train = dataset_size[0]
    num_label = dataset_size[2]
    adjlist = sp.lil_matrix((num_label, num_label), dtype=np.int32)
    sum_y = np.zeros(num_label, dtype=np.int32)

    for i in range(num_train):
        label_list = f.readline().split(" ")[0].split(",")
        for i, ind1 in enumerate(label_list):
            sum_y[int(ind1)] += 1
            for ind2 in label_list[i+1:]:
                adjlist[int(ind1), int(ind2)] += 1
                adjlist[int(ind2), int(ind1)] += 1

    print("make adjlist: end")
    f.close()
    return adjlist.tocsr(), sum_y

# large scale knn graph
def large_adjlist_k(embedding_method, k, train_file_name, output_file_name=""):
    adjlist, sum_y = make_sp_adjlist(train_file_name)
    num_label = adjlist.shape[0]
    max_k = range(0, k)

    if embedding_method == "line" or embedding_method == "node2vec":
        f = open(output_file_name + "_k" + str(k) + ".tsv", "a")
        for ind1 in range(num_label):
            print(ind1)
            indices = np.argsort(-adjlist.getrow(ind1).toarray()[0])[max_k]
            for ind2 in indices:
                if adjlist[ind1, ind2] > 0:
                    f.write(str(ind1) + "\t" + str(ind2) + "\n")
        f.close()

    elif embedding_method == "deepwalk":
        f = open(output_file_name + "_k" + str(k) + ".adjlist", "a")
        for ind1 in range(num_label):
            f.write(str(ind1))
            indices = np.argsort(-adjlist.getrow(ind1).toarray()[0])[max_k]
            for ind2 in indices:
                if adjlist[ind1, ind2] > 0:
                    f.write(" " + str(ind2))
            f.write("\n")
        f.close()

if __name__ == '__main__':
    # large_adjlist_k(embedding_method="line", k=5,
    #                 train_file_name="./dataset/Amazon/amazon_train.txt",
    #                 output_file_name="./m_data/Amazon/amazon")
    # adjlist_p(embedding_method="line", k=20, pmi_min=0.0001,
    #               train_file_name="./dataset/Wiki10/wiki10_train.txt",
    #               output_file_name="./m_data/Wiki10/wiki10")
    # adjlist_w(embedding_method="line", k=20, weight_min=0.2,
    #               train_file_name="./dataset/Wiki10/wiki10_train.txt",
    #               output_file_name="./m_data/Wiki10/wiki10")
    # adjlist_k(embedding_method="line", k=20,
    #               train_file_name="./dataset/Wiki10/wiki10_train.txt",
    #               output_file_name="./m_data/Wiki10/wiki10")
    # adjlist_k(embedding_method="deepwalk", k=20,
    #               train_file_name="./dataset/Wiki10/wiki10_train.txt",
    #               output_file_name="./m_data/Wiki10/wiki10")
    adjlist_k(embedding_method="line", k=40,
                  train_file_name="./dataset/DeliciousLarge/deliciousLarge_train.txt",
                  output_file_name="./m_data/DeliciousLarge/deliciousLarge")
    # adjlist_k(embedding_method="line", k=20,
    #               train_file_name="./dataset/AmazonCat/amazonCat_train.txt",
    #               output_file_name="./m_data/AmazonCat/amazonCat")
    # adjlist_k(embedding_method="line", k=5,
    #           train_file_name="./dataset/Amazon/amazon_train.txt",
    #           output_file_name="./m_data/Amazon/amazon")