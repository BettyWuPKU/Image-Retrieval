# 这是BoW模型的测试代码

import matplotlib.image as mpimg
import cv2
import numpy as np
import math
from sklearn.cluster import KMeans
import random
from display_img import show_pic
import time
sift = cv2.xfeatures2d.SIFT_create(nfeatures=500)
dataset_generated = True

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

@timeprinter('getClusterCenter')
def getClusterCenters(img_paths, img_data, num_words):
    '''
    得到所有图像的sift特征，并且通过k聚类得到k个单词
    To get the sift features of each pic and get the words of the vocabulary
    :param img_paths: the list of the paths of all the images
    :param img_data: if the input is already data then use this way to compute
    :param num_words: the number of the cluster center
    :return: the sift feature list and the center for the cluster
    '''
    begin_index = [0] # 保存前i个图像的特征向量的总数量
    des_list = []
    des_matrix = np.zeros((1, 128))
    gcc_cnt = 0
    # 读取所有图像，得到所有图像的特征的向量des，合并成一个大的特征的矩阵des_matrix(des_num*128规模)
    # gcc_cnt是在所有图像中该图像的des在des_matrix中的开始的序号
    # des_list是一个有img_num个list的list，第i个list就是第i个图像的sift特征向量的list
    print('Extracting features...')
    if img_paths != None:
        for path in img_paths:#the list of the paths of all the images
            im = mpimg.imread(path)
            kp, des = sift.detectAndCompute(im, None) #des 500*128
            des_matrix = np.row_stack((des_matrix, des))
            gcc_cnt += des.shape[0]
            begin_index.append(gcc_cnt)
            des_list.append(des)
    elif img_data != None:
        for im in range(img_data.shape[0]):
            kp, des = sift.detectAndCompute(im, None)
            des_matrix = np.row_stack((des_matrix, des))
            gcc_cnt += des.shape[0]
            begin_index.append(gcc_cnt)
            des_list.append(des)
    else:
        raise ValueError('illegal input')
    print('Features extracted.')
    des_matrix = des_matrix[1:, :]

    # 得到k个单词，kcenters是word_nums*128规模的矩阵，每一行是一个word的向量
    kmeans = KMeans(n_clusters=num_words, random_state=9)
    print('Forming words...')
    kmeans.fit(des_matrix)
    kcenters = kmeans.cluster_centers_
    # 返回所有图像的des特征list，单词，所有特征的标签，和图像的des在所有des的list中的起始下标
    print('Words formed.')
    return des_list, kcenters, kmeans.labels_, begin_index

@timeprinter('des2feature')
def des2features(des, num_words, centers, k, begin_index, labels):
    '''
    Transform the sift features into the vocabulary
    :param des: the sift features of one image(n * 128)(n: the number of features)
    :param num_words: the number of the cluster center
    :param centers: cluster centers or the words
    :param k: the kth image in the array(begin from 0)
    :return: the feature vector of the image(1 * num_words)
    '''
    # 所谓的构建词典，就是对某一个图像，统计其中sift特征中属于每个word的数量
    # 可以通过计算每个des和word的欧氏距离找最近的word，但直接使用聚类时产生的label会更快
    # labels[i]表示在des_matrix中第i个des的label，而根据之前记录的每个图像的des的下标可以直接算得
    # 第k个图像的第i个特征属于哪一个word
    print('Calculating words for each image...')
    img_feature_vec = np.zeros((1, num_words), 'float32')
    if(k != None):#统计某一张图的单词个数，训练版本，图库中的图
        for i in range(des.shape[0]): # the number of sift features in training img
            # 特征是一个1*word_num的向量，即每个word在图片中的数量
            img_feature_vec[0][labels[begin_index[k] + i]] += 1
    else:
        # 对于query的图像只能通过欧氏距离计算word的数量，查询版本，可能不是图库中的图
        for i in range(des.shape[0]): # the number of sift features in query img
            feature = des[i] # 第i个特征
            feature_k_rows = np.ones((num_words, 128), 'float32') # 初始化一个num_words * 128的1矩阵
            feature_k_rows = feature_k_rows * feature # 将feature拷贝num_words行
            dis2words = np.sum((feature_k_rows - centers) ** 2, 1) # 将该特征与每个word求欧氏距离(平方)
            index = np.argmin(dis2words) # 返回距离最小的index(但参考代码里不知道为什么是最大??)
            img_feature_vec[0][index] += 1
    print('Calculated.')
    return img_feature_vec

@timeprinter('getNearest')
def getNearest(feature, feature_dataset, num_close):
    '''
    get the num_close nearest images of the input feature of the query image
    :param feature: the feature of the query image
    :param feature_dataset: the features of the dataset
    :param num_close: the number of images we need to get
    :return: the indexes of the nearest images
    '''
    # 实际上就是一个和feature_dataset大小一样的1矩阵(img_num * cluster_num)
    features = np.ones((feature_dataset.shape[0], len(feature)), 'float32')
    features = features * feature
    # 计算query图像的feature和所有图像的feature之间的欧氏距离
    dis_img2dataset = np.sum((feature - feature_dataset) ** 2, 1)
    index = np.argsort(dis_img2dataset) # 将其按照距离从小到大排序后的索引存入index
    return index[:num_close]


# 暴力搜索...
def retrieveImage(img_path, dataset, centers, num_close, num_words):
    img = mpimg.imread(img_path)

    kp, des = sift.detectAndCompute(img, None)
    feature = des2features(des, num_words, centers)
    ret_index = getNearest(feature, dataset, num_close)

    return ret_index

@timeprinter('calcTF_IDF')
def calcTF_IDF(feature_dataset):
    # 使用TF_IDF的原因，TF反映了每个图像中某个单词出现的频率，IDF反映了某个单词在所有图像中出现的概率
    # 当TF高的时候说明该word可以很好的代表这个图像；但是如果IDF低则说明这个word在很多图像中有，并不具有代表性
    # 因此需要两个指标相结合；TF越高，IDF越高是越好的，这里直接相乘
    num_img = feature_dataset.shape[0]
    num_words = feature_dataset.shape[1]
    tf_idf = np.zeros((num_img, num_words), 'float32')
    idf = np.zeros(num_words, 'float32')
    # 在第i个图像中j单词出现的概率
    for i in range(num_img):
        f_sum = np.sum(feature_dataset[i])
        for j in range(num_words):
            tf_idf[i][j] = feature_dataset[i][j] / f_sum
    # 出现j单词的文件数量越多，idf越小
    for j in range(num_words):
        j_word_sum = 0
        for i in range(num_img):
            if feature_dataset[i][j] != 0: j_word_sum += 1
            # +1是为了防止分母为0
        idf[j] = math.log(num_img / (j_word_sum + 1e-6))
    # 计算每个图像的j单词的tf-idf
    for i in range(num_img):
        for j in range(num_words):
            tf_idf[i][j] = tf_idf[i][j] * idf[j]
    # 返回的tf_idf即是最后计算出来的图像的feature'，其实idf相当于对每个图像中的tf进行加权
    return tf_idf, idf


@timeprinter('retrieveImageTF_IDF')
def retrieveImageTF_IDF(img_path,query_num, dataset, centers, num_close, num_words):
    print('Querying ' + img_path)
    if query_num != -1:
        img = mpimg.imread(img_path)
        kp, des = sift.detectAndCompute(img, None)
    else:
        des = des_list[query_num]
    feature = des2features(des, num_words, centers, query_num, begin_index, labels)
    tf_idf_dataset, idf = calcTF_IDF(dataset)
    tf_idf_query = np.zeros(num_words, 'float32')
    f_sum = np.sum(feature)
    # 计算query图像的tf_idf值
    for j in range(num_words):
        tf_idf_query[j] = feature[0][j] / f_sum * idf[j]
    ret_index = getNearest(tf_idf_query, tf_idf_dataset, num_close)

    return ret_index


# 尚未使用，但是应该能加快速度
def inverseIndex(num_image, num_words, labels, begin_index):
    words = []
    for i in range(num_words):
        words.append('Label_' + str(i))
    inversed_index = dict.fromkeys(words, '')
    for i in range(num_words):
        inversed_index[words[i]] = []
    for i in range(num_image):
        id = begin_index[i]
        total_num = begin_index[i + 1] - id
        for j in range(total_num):
            # print(inversed_index[words[labels[id]]])
            inversed_index[words[labels[id]]].append([i, j]) # 第i个文件的第j个特征属于这一类
            id += 1
    return inversed_index


#可变参数
image_num = 200
query_num = random.randint(0, image_num - 1)
center_num = 30

input_paths = []
path_pattern = '../Images/ukbench/full/ukbench'
query_path = path_pattern + str(query_num).zfill(5) + '.jpg'
for i in range(image_num):
    path = path_pattern + str(i).zfill(5) + '.jpg'
    input_paths.append(path)

des_list, centers, labels, begin_index = None, None, None, None
if dataset_generated == False:
    des_list, centers, labels, begin_index = getClusterCenters(input_paths, None, center_num)
    # list array array list
    np.save('./data/des_list.npy', np.array(des_list))
    np.save('./data/begin_index.npy', np.array(begin_index))
    centers.tofile('./data/centers.bin')
    labels.tofile('./data/labels.bin')
else:
    des_list = np.load('./data/des_list.npy', allow_pickle=True).tolist()
    begin_index = np.load('./data/begin_index.npy', allow_pickle=True).tolist()
    centers = np.fromfile('./data/centers.bin', dtype='float64')
    centers = np.reshape(centers, (center_num, 128))
    labels = np.fromfile('./data/labels.bin', dtype='int32')
    labels = np.reshape(labels, begin_index[image_num])

# 统计所有图片中特征的单词数量
allFeatureVec = np.zeros((len(des_list), center_num), 'float32')
if dataset_generated == False:
    for i in range(image_num):
        allFeatureVec[i] = des2features(des_list[i], center_num, centers, i, begin_index, labels)
    allFeatureVec.tofile('./data/img_dataset.bin')
else:
    allFeatureVec = np.fromfile('./data/img_dataset.bin', dtype='float32')
    allFeatureVec = np.reshape(allFeatureVec, (len(des_list), center_num))

ranking = retrieveImageTF_IDF(query_path,query_num, allFeatureVec, centers, image_num, center_num)

#print(ranking)
result = [input_paths[i] for i in ranking]
show_pic(N=3, M=3, img_paths=result[:6], query_path=query_path)
