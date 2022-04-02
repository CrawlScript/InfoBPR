# InfoBPR
Simple Yet Powerful Ranking Loss

We provide InfoBPR API for both TensorFlow and PyTorch.

## Paper
MGDCF: Distance Learning via Markov Graph Diffusion for Neural Collaborative Filtering




## Performance

MF (Matrix Factorization) can beat state-of-the-art GNN-based CF approaches with our InfoBPR Loss (MF-InfoBPR).

<p align="center">
<img src="plots/Yelp2018.png" width="500"/>
</p>


InfoBPR also can improve the optimization of CF models.

<p align="center">
<img src="plots/InfoBPR_optimization.png" width="700"/>
</p>

## Installation


```bash
pip install info_bpr
```

Note that the installed info_bpr library support both TensorFlow and PyTorch.



## Usage

For PyTorch users:

```python
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
```



## DEMO

[Matrix Factorization with InfoBPR Loss (MF-InfoBPR)](demo/demo_torch_mf_info_bpr.py)