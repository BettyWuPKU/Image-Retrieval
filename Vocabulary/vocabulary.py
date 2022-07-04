import numpy as np
from sklearn.cluster import MiniBatchKMeans
import networkx as nx
import cv2
import math
import pickle
import time

# 当图像在文件夹中的顺序和自然数不是一一映射的时候？需要建立一个映射关系
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
    with open('obj/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


class VocabularyTree(object):
    def __init__(self, branch_num, depth, dataset, imgnum, descriptor='Sift'):
        # dataset[i]给出第i副图像的路径
        self.branch_num = branch_num
        self.depth = depth
        self.imgnum = imgnum

        # 每个结点是其子结点的中心，一个feature，按照树的前序周游编号
        # 是一个字典，key为i对应的value就是i号node的feature值
        self.nodes = {}
        # 记录了每个feature 128维向量和index的对应
        self.tree = {}
        # 初始化一个有向图
        self.graph = nx.DiGraph()

        self.current_index = 0
        # 记录该图像是否被统计过在结点上的分布
        self.visited = set()
        self.dataset = dataset
        # 检索时的weight，weight[i]=log(N/N[i]);表示i结点的权重=log(总图像数/经过该结点的图像数)
        # 类似于TF-IDF方法中的IDF，越多图像经过，越低表明越没有区分度
        self.retrieve_weight = {}
        # self.graph.nodes[node][imgID]记录某个图像经过该结点的次数，imgID的图像经过node结点的次数
        self.dataset_vec = [[] for i in range(imgnum)]
        # 根据论文中计算vec的方式定义，n[i][j]为图像i经过特征j的次数；n[i]就是图像i的频率的向量
        # q[i]为n[i]点乘weight之后的结果，是图像i在vec
        self.n = []
        self.q = []
        self.q_norm = []

    def self_mser_extract_features(self, img):
        mser = cv2.MSER_create()
        regions, boxes = mser.detectRegions(img)
        kpkp = mser.detect(img)
        for i in range(len(boxes)):
            x, y, w, h = boxes[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        sift = cv2.xfeatures2d.SIFT_create()
        des = sift.compute(img, kpkp)
        return des

    # 提取图像的特征
    def sift_extract_features(self, img):
        sift = cv2.xfeatures2d.SIFT_create(nfeatures=500)
        # des = sift.compute(img, kpkp)
        kp, des = sift.detectAndCompute(img, None)
        return des

    # 建树，将所有的特征提取出来之后层次聚类
    def build(self):
        features = self.extract_features()
        # np.save('features.npy', features)
        self.divide(features)
        return self

    # 将所有的特征提取出来

    @timeprinter('extract_features')
    def extract_features(self):
        print('Extracting features...')
        # 存储每个图像feature的数量
        feature_len = []
        features = np.zeros((1, 128))
        for id in range(self.imgnum):
            if (id % 100 == 99):
                print('Extacted 100 images')
            im = self.get_image(id)
            des = self.sift_extract_features(im)
            feature_len.append(len(des))
            features = np.row_stack((features, des))
        print('Extraction completed')
        # np.save('features_len.npy', feature_len)
        # np.save('features.npy', features)
        return features[1:, :]


    def divide(self, features, node=0, root=None, current_depth=0):
        if root is None:
            root = np.mean(features, axis=0)

        self.nodes[node] = root
        self.graph.add_node(node)

        # 如果是树叶结点(到达要求的深度或者是无法进行聚类了)则返回
        if current_depth >= self.depth or len(features) < self.branch_num:
            return

        cluster_model = MiniBatchKMeans(n_clusters=self.branch_num)
        cluster_model.fit(features)

        child_node = [[] for i in range(self.branch_num)]
        # 将该结点为聚类中心的所有的feature分配到子结点
        for i in range(len(features)):
            child_node[cluster_model.labels_[i]].append(features[i])

        self.tree[node] = []
        for i in range(self.branch_num):
            self.current_index += 1
            # 记录了每个feature 128维向量和index的对应
            self.tree[node].append(self.current_index)
            self.graph.add_edge(node, self.current_index)
            self.divide(child_node[i], self.current_index, cluster_model.cluster_centers_[i], current_depth + 1)

        return

    def get_image(self, imgID):
        img = cv2.imread(self.dataset[imgID])
        return img

    # 将数据集或者是查询图像的特征在树上检索，记录每次检索的path
    def retrieve_leaf(self, imgID):
        if imgID in self.visited:
            return
        img = self.get_image(imgID)
        features = self.sift_extract_features(img)

        for feature in features:
            node = 0
            path = [0]
            while self.graph.out_degree(node):
                max_sim = -float("inf")
                closest = None
                for child in self.graph[node]:
                    # 点积表示相似度
                    sim = np.linalg.norm([self.nodes[child] * feature])
                    # 或者用距离衡量：distance = np.linalg.norm([self.nodes[child] - feature])
                    # 还需要把max_sim初始化为min_sim=float('inf')
                    if sim > max_sim:
                        max_sim = sim
                        closest = child
                path.append(closest)
                node = closest

            for i in range(len(path)):
                node = path[i]
                if imgID not in self.graph.nodes[node]:
                    # nodes[node][imgID]记录了一类特征在imgID的图像中出现的次数
                    self.graph.nodes[node][imgID] = 1
                else:
                    self.graph.nodes[node][imgID] += 1

        self.visited.add(imgID)
        return


    def calc_retrieve_weight_on_node(self):
        for imgID in range(self.imgnum):
            self.retrieve_leaf(imgID)
        for node in self.nodes:
            n = self.graph.nodes[node]
            self.retrieve_weight[node] = math.log2(self.imgnum / (1e-5 + len(n)))

    # 计算每个图像在结点上的
    def calc_dataset_vec(self):
        self.n = []
        self.q = []
        self.q_norm = []
        for imgID in range(self.imgnum):
            n_imgID = []
            q_imgID = []
            for i in self.nodes:
                # n_i表示imgID经过i号结点的特征数
                n_i = 0
                if imgID in self.graph.nodes[i]:
                    n_i = self.graph.nodes[i][imgID]
                n_imgID.append(n_i)
                q_imgID.append(n_i * self.retrieve_weight[i])
            self.n.append(n_imgID)
            self.q.append(q_imgID)
            self.q_norm.append(np.linalg.norm(q_imgID, ord=2, axis=None, keepdims=False))

    @timeprinter('Building up vocabulary')
    def prev_work(self, dataset):
        print('Building up vocabulary...')
        self.build()
        print('Vocabulary built.')
        print('Calculating the node weight and dataset images vectors...')
        self.calc_retrieve_weight_on_node()
        self.calc_dataset_vec()
        print('Finish all the prev works, now you can retrieve.')

    def calc_query_vec(self, imgID):
        d = []
        self.retrieve_leaf(imgID)
        for i in self.nodes:
            if imgID in self.graph.nodes[i]:
                d.append(self.graph.nodes[i][imgID] * self.retrieve_weight[i])
            else:
                d.append(0)
        return d

    # retrieve the most similar images and return the ranking
    def image_retrieval(self, query_path):
        self.dataset.append(query_path)
        d = self.calc_query_vec(self.imgnum)
        dis = []
        d_norm = np.linalg.norm(d, ord=2, axis=None, keepdims=False)
        for i in range(self.imgnum):
            dis_i = self.q[i] / self.q_norm[i] - d / d_norm
            dis_i = np.linalg.norm(dis_i, ord=2, axis=None, keepdims=False)
            dis.append(dis_i)

        return np.array(dis).argsort()

    def save(self):
        basic_parameters = [self.branch_num, self.depth, self.imgnum, self.current_index]
        save_obj(basic_parameters, 'basic_parameters')
        save_obj(self.nodes, 'nodes')
        save_obj(self.tree, 'tree')
        save_obj(self.graph, 'graph')
        save_obj(self.visited, 'visited')
        save_obj(self.dataset, 'dataset')
        save_obj(self.retrieve_weight, 'weight')
        save_obj(self.n, 'n')
        save_obj(self.q_norm, 'q_norm')
        save_obj(self.q, 'q')

        # tree_inform = [self.nodes, self.tree, self.graph, self.visited, self.dataset,
        #                self.retrieve_weight, self.dataset_vec, self.n, self.q, self.q_norm]

    def load(self):
        basic_parameters = load_obj('basic_parameters')
        self.branch_num, self.depth, self.imgnum, self.current_index = basic_parameters
        self.nodes = load_obj('nodes')
        self.tree = load_obj('tree')
        self.graph = load_obj('graph')
        self.visited = load_obj('visited')
        self.dataset = load_obj('dataset')
        self.retrieve_weight = load_obj('weight')
        self.n = load_obj('n')
        self.q_norm = load_obj('q_norm')
        self.q = load_obj('q')











