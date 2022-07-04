import matplotlib.pyplot as plt
import numpy as np
import os
import random

def show_pic(N, M, img_paths, query_path):
    query_img = plt.imread(query_path)
    plt.subplot(N, M, N // 2)
    plt.imshow(query_img)
    plt.xticks([])
    plt.yticks([])

    cnt = M
    for path in img_paths:
        cnt += 1
        img = plt.imread(path)
        plt.subplot(N, M, cnt)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
    plt.show()

# dataset_path = 'oxbuild_images'
# dataset = [img_name for img_name in os.listdir(dataset_path)
#                             if not img_name.startswith('.')]
# query_path = random.sample(dataset, 1)
# query_path = dataset_path + '/' + query_path[0]
# query_img = plt.imread(query_path)
# plt.subplot(1, 2, query_img)
# plt.imshow(query_img)
# plt.xticks([])
# plt.yticks([])
