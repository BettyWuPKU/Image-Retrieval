import matplotlib.pyplot as plt

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


