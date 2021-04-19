# -*- coding:utf-8 -*-
# editor: zzh
# date: 2021/2/26

import codecs
from random import choice

# import numpy as np

# import keras.backend as K
# import tensorflow as tf

import pandas as pd

if __name__ == '__main__':

    test_data = pd.read_csv("..\\灾害类别分类\\DATAS\\DATA6\\test_data.csv")
    titles = test_data["title"]
    newss = test_data["news"]
    labels = test_data["label"]

    for t,l in zip(titles,labels):
        print(t,l)

    # data = np.random.random([2,10,2])
    # indices = np.random.randint(0,10,[2,5])
    #
    # print(data)
    # print(indices)
    #
    # with tf.Session() as sess:
    #     # print(K.gather(data,indices))
    #     print(tf.gather(params=data,indices=indices,batch_dims=1))
    #     d = sess.run(tf.gather(params=data,indices=indices,batch_dims=1))
    #     print(d)
