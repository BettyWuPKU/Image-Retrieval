from vocabulary import VocabularyTree
import os
import random
from display_img import show_pic
import time
branch_num = 6
depth = 6
image_num = 100
pre_load = True

datapath = '../Images/ukbench/full/ukbench'
dataset = []
# randomly generate the query image
query_path = '../Images/ukbench/full/ukbench'
query_num = random.randint(0, image_num - 1)
query_path += str(query_num).zfill(5) + '.jpg'
for i in range(image_num):
    dataset.append(datapath + str(i).zfill(5) + '.jpg')

Vocabulary = VocabularyTree(branch_num=branch_num, depth=depth, dataset=dataset, imgnum=image_num)

#Vocabulary.extract_features()
if (pre_load == False):
    Vocabulary.prev_work(dataset)
    isExists = os.path.exists('./obj')
    if not isExists:
        os.makedirs('./obj')
    Vocabulary.save()
else:
    isExists = os.path.exists('./obj')
    if not isExists:
        print('Data not saved.')
        exit()
    Vocabulary.load()


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

@timeprinter('query')
def query():
    print('Querying ukbench' + str(query_num).zfill(5) + '.jpg now')
    ranking = Vocabulary.image_retrieval(query_path=query_path)
    global result
    result = [Vocabulary.dataset[i] for i in ranking]
    #print(result)

query()
show_pic(N=3, M=3, img_paths=result[:6], query_path=query_path)
