# coding=utf-8
import os
import torch
from info_bpr.losses import th_info_bpr as info_bpr

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
a_embeddings = torch.randn(5, 10).cuda()
b_embeddings = torch.randn(5, 10).cuda()
edges = [[0, 1], [2, 4]]
from tqdm import tqdm
for _ in tqdm(range(1000)):
    loss = info_bpr(a_embeddings, b_embeddings, edges, num_negs=20)
print(loss)



