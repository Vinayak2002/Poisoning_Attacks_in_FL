def poison_data(labels: list):
    """
    Poisons the data by label shuffling.
    :param labels: List of labels to flip (poison).
    :return: Poisoned list.
    """

    d = {
        1: 7,
        2: 5,
        3: 8,
        4: 9,
        5: 8,
        6: 2,
        7: 1,
        8: 5,
        9: 4,
        0: 0,
    }

    pos_labels = []

    for i in labels:
        l = list(i)
        n = l.index(1)
        pos_labels.append(d[n])

    lb = LabelBinarizer()
    pos_label_list = lb.fit_transform(pos_labels)

    return pos_label_list
# # from numpy import array
# # from numpy import mean
# # from numpy import cov
# # from numpy.linalg import eig
# # # define a matrix
# # # A = array([[2, 3, 4, 5, 6, 7], [1, 5, 3, 6, 7, 8]])
# # # A = array([[1, 2], [3, 4], [5, 6]])
# # A = array([[2, 1], [3, 5], [4, 3], [5, 6], [6, 7], [7, 8]])
# # print(A)
# # # calculate the mean of each column
# # M = mean(A, axis=1)
# # print(M)
# #
# # # calculate covariance matrix
# # V = cov(A)
# # print(V)
# #
# # # eigendecomposition of covariance matrix
# # values, vectors = eig(V)
# #
# # print("Values: \n", values)
# # print()
# # print("Vectors: \n", vectors)
# # # print(values)
# # # project data
# # # P = vectors.T.dot(C.T)
# # # print(P.T)
# # from sklearn.decomposition import PCA
# #
# # pca = PCA(2)
# # # fit on data
# # pca.fit(A.T)
# # # access values and vectors
# # print(pca.components_)
# # print(pca.explained_variance_)
# # # transform data
# # B = pca.transform(A.T)
# # print(B)
# from sklearn.preprocessing import LabelBinarizer
#
# data = [1, 2, 1, 1, 2]
# import numpy
#
# d = {
#     1: 7,
#     2: 5,
#     3: 8,
#     4: 9,
#     5: 8,
#     6: 2,
#     7: 1,
#     8: 5,
#     9: 4,
#     0: 0,
# }
# nx = [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#       [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
#       [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
#       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
#       [0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
# dx = numpy.array(nx)
# p = []
# p1 = []
# for i in dx:
#     l = list(i)
#     n = l.index(1)
#     p1.append(n)
#     p.append(d[n])
#
# # print(p)
# # print(p1)
# lb = LabelBinarizer()
# label_list = lb.fit_transform(p)
#
# print(label_list)
#
# l1 = numpy.array(label_list[: 5])
# print()
# print(l1)
# numpy.random.shuffle(l1)
# print()
# print(l1)

n = "vinayak"
print(type(n))
s = str(type(n))
print("ok: " + s + " : ok")