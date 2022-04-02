# coding=utf-8
import os
import torch
from info_bpr.losses import th_info_bpr as info_bpr

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

num_users = 5
num_items = 5
embedding_size = 64
user_embeddings = torch.randn(num_users, embedding_size).cuda()
item_embeddings = torch.randn(num_items, embedding_size).cuda()
user_item_edges = [
    [0, 1],
    [0, 2],
    [2, 4],
    [3, 4]
]

ranking_loss = info_bpr(user_embeddings, item_embeddings, user_item_edges, num_negs=300)
print("InfoBPR Loss: ", ranking_loss)



