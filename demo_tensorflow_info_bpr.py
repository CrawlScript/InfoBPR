# coding=utf-8
import os
import tensorflow as tf
from info_bpr.losses import tf_info_bpr as info_bpr
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
a_embeddings = tf.random.truncated_normal([5, 64])
b_embeddings = tf.random.truncated_normal([5, 64])
edges = [[0, 1], [2, 4], [3, 4]]


info_bpr = tf.function(info_bpr)

for _ in tqdm(range(1000)):
    loss = info_bpr(a_embeddings, b_embeddings, edges, num_negs=20)

print(loss)



