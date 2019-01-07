import numpy as np
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
import time
import read_data as rd
import ClusteringData as cd


class LeXML(object):
    def __init__(self, train_file_name, test_file_name, rep_file_name, embedding_method="line", sparse=True, num_partition=2,
                 learning_rate=0.005, weight_decay=0.0001, dropout_rate=0.8, batch_size=128, epoch=10):
        self.num_partition = num_partition
        self.sparse = sparse
        if self.sparse:
            self.train_x, self.train_y = rd.get_sparse_data(train_file_name)
            self.test_x, self.test_y = rd.get_sparse_data((test_file_name))
        else:
            self.train_x,self.train_y = rd.get_data(train_file_name)
            self.test_x, self.test_y = rd.get_data((test_file_name))
        self.num_train = self.train_x.shape[0]
        self.num_test = self.test_x.shape[0]
        self.feature_dim = self.train_x.shape[1]
        self.label_dim = self.train_y.shape[1]
        self.rep_w = rd.get_rep_w(rep_file_name, embedding_method, self.label_dim)
        self.embedding_dim = self.rep_w.shape[1]
        if self.embedding_dim >= 256:
            self.m_dim = 512
        else:
            self.m_dim = 256

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.epoch = epoch

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape=shape, mean=0, stddev=0.01, dtype=tf.float32)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape, dtype=tf.float32)
        return tf.Variable(initial)

    def lexml(self):
        # --- Define feeding data operations ---
        self.dataX = tf.placeholder(dtype=tf.float32, shape=[None, self.feature_dim])
        self.dataY = tf.placeholder(dtype=tf.float32, shape=[None, self.label_dim])

        # --- Get embedding data ---
        embedding_x, w1, w2 = self.model(self.dataX)
        embedding_y = self.label_embedding(self.dataY)

        # --- Train and evaluate the model ---
        d_xy = tf.abs(tf.subtract(embedding_x, embedding_y))
        # d_xy = tf.square(tf.subtract(embedding_x, embedding_y))
        self.l2norm = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2)
        # self.l1norm = tf.reduce_sum((tf.abs(self.w)))
        self.loss = tf.reduce_sum(d_xy, [0,1]) + self.weight_decay * self.l2norm
        self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
        # self.train_op = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.loss)


        # --- Run tensorflow ---
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def model(self, input_x):
        self.keep_prob = tf.placeholder(tf.float32)
        h1 = tf.nn.l2_normalize(input_x, 1)
        w1 = self.weight_variable([self.feature_dim, self.m_dim])
        b1 = self.bias_variable([self.m_dim])
        h2 = tf.matmul(h1, w1) + b1
        # h2 = tf.matmul(h1, w1, a_is_sparse=self.sparse) + b1
        h3 = tf.nn.relu(h2)
        h3_drop = tf.nn.dropout(h3, self.keep_prob)

        w2 = self.weight_variable([self.m_dim, self.embedding_dim])
        b2 = self.bias_variable([self.embedding_dim])
        h4 = tf.matmul(h3_drop, w2) + b2
        h4_drop = tf.nn.dropout(h4, self.keep_prob)
        self.output_x = tf.nn.l2_normalize(h4_drop, 1)
        return self.output_x, w1, w2

    def label_embedding(self, input_y):
        u = tf.constant(self.rep_w, dtype=tf.float32)
        fy = tf.matmul(input_y, u)
        # fy = tf.matmul(input_y, u, a_is_sparse=self.sparse)
        self.output_y = tf.nn.l2_normalize(fy, 1)
        return self.output_y

    def calc_embedding_x(self, x):
        return self.sess.run(self.output_x, feed_dict={self.dataX: x, self.keep_prob: 1.0})

    def return_embedding_val(self, data_x, step=1024):
        num_data = data_x.shape[0]
        embedding_x = np.zeros((num_data, self.embedding_dim))
        for i in range(0, num_data, step):
            j = i + step - 1 if i + step - 1 < num_data else num_data
            if self.sparse:
                data_sparse_x = np.array([data_x.getrow(row).toarray()[0] for row in range(i,j)])
                embedding_x[i:j] = self.calc_embedding_x(data_sparse_x)
            else:
                embedding_x[i:j] = self.calc_embedding_x(data_x[i:j])
        return embedding_x

    def training(self):
        start_time = time.time()
        for i in range(self.epoch):
            print("-- " + str(i) + " epoch -----")

            sff_ind = np.random.permutation(self.num_train)
            for ind in range(0, self.num_train, self.batch_size):
                indices = sff_ind[ind: ind + self.batch_size if ind + self.batch_size < self.num_train else self.num_train]
                if self.sparse:
                    batch_x = np.array([self.train_x.getrow(row).toarray()[0] for row in indices])
                    batch_y = np.array([self.train_y.getrow(row).toarray()[0] for row in indices])
                else:
                    batch_x = self.train_x[indices]
                    batch_y = self.train_y[indices]

                _, loss_val, w_loss = self.sess.run([self.train_op, self.loss, self.l2norm], feed_dict={self.dataX: batch_x, self.dataY: batch_y, self.keep_prob: self.dropout_rate})
                if ind == 0:
                    print("loss:   " + str(loss_val))
                    print("w_norm: " + str(w_loss))

        training_time = time.time() - start_time
        print("train time: {:.2f}[s]".format(training_time))

    def accuracy(self, predict):
        count = 0
        for i in range(0, 5):
            for j in range(self.num_test):
                if self.test_y[j, int(predict[j][i])] == 1:
                    count += 1
            if i % 2 == 0:
                print("P@" + str(i+1) + " accuracy: {:.8f}".format(count / (self.num_test * (i+1))))

    def setup_knn(self, embedding_train_x, n_neighbors=30):
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(embedding_train_x, self.train_y)
        return knn

    def setup_cls_knn(self, embedding_train_x, n_neighbors=30):
        start_time = time.time()
        cls, c_predict, centers = cd.clustering(embedding_train_x, self.num_partition)
        cls_time = time.time() - start_time
        print("cls time: {:.2f}[s]".format(cls_time))
        return cd.cls_partition(embedding_train_x, self.train_y, cls, c_predict, centers, self.num_partition, n_neighbors, self.sparse)

    def predict_by_bruteforce(self, knn, embedding_test_x, n_neighbors=30, k_pred=5):
        predict = np.zeros((self.num_test, 5))
        max_k = range(0, k_pred)

        start_time = time.time()
        indices = knn.kneighbors(embedding_test_x, return_distance=False)
        knns_time = time.time() - start_time
        print("knns time: {:.2f}[s]".format(knns_time))
        for k in range(5, n_neighbors + 1, 5):
            print("-- knn: " + str(k) + " ---------------")
            start_time = time.time()
            for i in range(self.num_test):
                sum_y = np.sum(self.train_y[indices[i, 0:k], :], axis=0)
                if self.sparse:
                    predict[i] = np.argsort(-sum_y)[0, max_k]
                else:
                    predict[i] = np.argsort(-sum_y)[max_k]

            predict_time = time.time() - start_time
            self.accuracy(predict)
            print("pred time: {:.2f}[s]".format(predict_time))

    def predict_by_clustering(self, Z, embedding_test_x, n_neighbors=30, k_pred=5):
        predict = np.zeros((self.num_test, 5))

        for k in range(15, n_neighbors + 1, 5):
            print("-- knn: " + str(k) + " ---------------")
            start_time = time.time()
            for test_ind, test_data in enumerate(embedding_test_x):
                dist = np.zeros(self.num_partition)
                for i in range(self.num_partition):
                    dist[i] = np.linalg.norm(test_data - Z[i].return_center())
                c = np.argmin(dist)
                predict[test_ind] = Z[c].return_predict(test_data.reshape(1, -1), k, k_pred)

            predict_time = time.time() - start_time
            self.accuracy(predict)
            print("pred time: {:.2f}[s]".format(predict_time))

    def eval(self, use_clustering=False):
        print("-- predict -----")
        embedding_train_x = self.return_embedding_val(self.train_x)

        start_time = time.time()
        embedding_test_x = self.return_embedding_val(self.test_x)
        embedding_time = time.time() - start_time
        print("emb time:  {:.2f}[s]".format(embedding_time))

        if use_clustering:
            knn = self.setup_cls_knn(embedding_train_x)
            self.predict_by_clustering(knn, embedding_test_x)
        else:
            Z = self.setup_knn(embedding_train_x)
            self.predict_by_bruteforce(Z, embedding_test_x)

    def auto_learning(self, num_loop=20, use_clustering=False):
        self.lexml()
        for loop in range(num_loop):
            print("LOOP: " + str(loop))
            self.training()
            self.eval(use_clustering)


# debug
if __name__ == "__main__":
    model = LeXML("dataset/Wiki10/wiki10_train.txt",
                  "dataset/Wiki10/wiki10_test.txt",
                  "m_data/Wiki10/wiki10_li300_k20_XS1.embeddings",
                  embedding_method="line", sparse=False, num_partition=2)

    # model = LeXML("dataset/Wiki10/wiki10_train.txt",
    #               "dataset/Wiki10/wiki10_test.txt",
    #               "m_data/Wiki10/wiki10_nv300_k20_2.embeddings",
    #               embedding_method="node2vec", sparse=False, num_partition=2)

    # model = LeXML("dataset/Wiki10/wiki10_train.txt",
    #               "dataset/Wiki10/wiki10_test.txt",
    #               "m_data/Wiki10/wiki10_dw300_k20_1.embeddings",
    #               embedding_method="deepwalk", sparse=False, num_partition=2)

    # model = LeXML("dataset/DeliciousLarge/deliciousLarge_train.txt",
    #               "dataset/DeliciousLarge/deliciousLarge_test.txt",
    #               "m_data/DeliciousLarge/deliciousLarge_",
    #               embedding_method="line", sparse=True, num_partition=2)

    model.auto_learning(num_loop=20, use_clustering=False)