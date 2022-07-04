import cv2
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA as sklearnPCA
from display_img import show_pic
import random
import pickle
import time

PCA_DIM = 128
VLAD_LEN = 32

def timeprinter(str):
    def special_timeprinter(func):
        def wrapped(*args):
            t1  = time.time()
            res = func(*args)
            t2 = time.time()
            print(f"{str} takes {t2-t1} seconds")
            return res
        return wrapped
    return special_timeprinter

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def extract_feature(path):
    img = cv2.imread(path)
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=500)
    kp, des = sift.detectAndCompute(img, None)

    return des

@timeprinter('get_cluster_center')
def get_cluster_center(des, K):
    kmeans = KMeans(n_clusters=K, random_state=9)
    kmeans.fit(des)
    kcenters = kmeans.cluster_centers_
    labels = kmeans.labels_
    return kcenters, labels

def get_des(img_paths):
    des_matrix = np.zeros((1, 128))
    des_len = []
    for path in img_paths:
        des = extract_feature(path)
        des_matrix = np.row_stack((des_matrix, des))
        des_len.append(len(des))
    des_matrix = des_matrix[1:]
    return des_matrix, des_len

@timeprinter('des2vlad')
def des2vlad(des_matrix, des_len, labels, centers):
    vlad_list_pca = []
    vlad_list = []
    cnt = 0

    for l in des_len:
        des = des_matrix[cnt:cnt + l]
        label = labels[cnt:cnt + l]

        vlad = np.zeros((VLAD_LEN, 128))
        for i in range(l):
            vlad[label[i]] += (des[i] - centers[label[i]])
        # vlad_norm = cv2.normalize(vlad, vlad, 1.0, 0.0, cv2.NORM_L2)
        vlad = vlad.reshape(VLAD_LEN * 128, -1)
        vlad_list.append(vlad)
        cnt += l
    vlad_list = np.array(vlad_list)
    vlad_list = vlad_list.reshape(-1, VLAD_LEN * 128)
    PCA = sklearnPCA(n_components=PCA_DIM)
    vlad_pca = PCA.fit_transform(vlad_list)
    vlad_pca_norm = vlad_pca.copy()
    for vlad, vlad_norm in zip(vlad_pca, vlad_pca_norm):
        cv2.normalize(vlad, vlad_norm, 1.0, 0.0, cv2.NORM_L2)
    return vlad_pca_norm, PCA


def calc_dist(d1, d2):
    # l2 norm
    return np.linalg.norm(d1 - d2)


def get_query_vlad(des_list, des_len, centers, PCA):
    vlad = np.zeros((VLAD_LEN, 128))
    for i in range(des_len):
        min_dis = float('inf')
        closest = 0
        for j in range(VLAD_LEN):
            dis = calc_dist(des_list[i], centers[j])
            if(dis < min_dis):
                min_dis = dis
                closest = j
        vlad[closest] += des_list[i] - centers[closest]
    vlad = vlad.reshape(-1, VLAD_LEN * 128)
    # PCA = sklearnPCA(n_components=PCA_DIM)
    vlad_pca = PCA.transform(vlad)
    vlad_pca_norm = vlad_pca.copy()
    cv2.normalize(vlad_pca, vlad_pca_norm, 1.0, 0.0, cv2.NORM_L2)
    return vlad_pca_norm


def retrieve(img_paths, query_paths):
    des_matrix, des_len = get_des(img_paths)
    des_q, des_len_q = get_des(query_paths)
    centers, labels = get_cluster_center(des_matrix, VLAD_LEN)
    vlad_list_pca, pca = des2vlad(des_matrix, des_len, labels, centers)

    cnt = 0
    query_vlad_list_pca = []
    for i in range(len(query_paths)):
        des = des_q[cnt:cnt + des_len_q[i]]
        vlad_pca_q = get_query_vlad(des, des_len_q[i], centers, pca)
        query_vlad_list_pca.append(vlad_pca_q)
        cnt += des_len_q[i]

    for i in range(len(query_paths)):
        print('Querying' + query_paths[i])
        dist_list = []
        for j in range(len(img_paths)):
            dist = calc_dist(query_vlad_list_pca[i], vlad_list_pca[j])
            dist_list.append(dist)

        sim = np.array(dist_list)
        ranking = sim.argsort()
        result = [img_paths[k] for k in ranking]

        show_pic(N=3, M=3, img_paths=result[:6], query_path=query_paths[i])

dataset_generated = True
data_path = "../Images/ukbench/full/ukbench"
img_paths = []
img_num = 130
# PCA降维需要样本空间大小>=维度
assert img_num >= VLAD_LEN
for i in range(img_num):
    path = data_path + str(i).zfill(5) + '.jpg'
    img_paths.append(path)

query_id = [random.randint(0, img_num - 1)]
query_paths = []
for id in query_id:
    path = data_path + str(id).zfill(5) + '.jpg'
    query_paths.append(path)

des_matrix, des_len, centers, labels, vlad_list_pca, pca = [None for i in range(6)]

if (dataset_generated == False):
    des_matrix, des_len = get_des(img_paths)
    centers, labels = get_cluster_center(des_matrix, VLAD_LEN)
    vlad_list_pca, pca = des2vlad(des_matrix, des_len, labels, centers)
    save_obj([des_matrix, des_len, centers, labels, vlad_list_pca, pca], 'VLAD_data')
else:
    isExists = os.path.exists('./VLAD_data.pkl')
    if not isExists:
        print('Error! No data saved.')
        exit()
    des_matrix, des_len, centers, labels, vlad_list_pca, pca = load_obj('VLAD_data')

# retrieve(img_paths, query_paths)
des_q, des_len_q = get_des(query_paths)
cnt = 0
query_vlad_list_pca = []
for i in range(len(query_paths)):
    des = des_q[cnt:cnt + des_len_q[i]]
    vlad_pca_q = get_query_vlad(des, des_len_q[i], centers, pca)
    query_vlad_list_pca.append(vlad_pca_q)
    cnt += des_len_q[i]

@timeprinter('query')
def query():
    for i in range(len(query_paths)):
        print('Querying' + query_paths[i])
        dist_list = []
        for j in range(len(img_paths)):
            dist = calc_dist(query_vlad_list_pca[i], vlad_list_pca[j])
            dist_list.append(dist)

        dist_list = np.array(dist_list)
        ranking = dist_list.argsort()
        global result
        result = [img_paths[k] for k in ranking]

query()
show_pic(N=3, M=3, img_paths=result[:6], query_path=query_paths[i])
