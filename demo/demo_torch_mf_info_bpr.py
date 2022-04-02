# coding=utf-8
import os

from grecx.evaluation import evaluate_mean_global_metrics
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import numpy as np

from info_bpr.losses import th_info_bpr as info_bpr

# require tensorflow (cpu or gpu)
import tensorflow as tf
import grecx
from grecx.datasets.light_gcn_dataset import LightGCNDataset

device = "cuda"

dataset = "light_gcn_yelp"  # "light_gcn_yelp" | "light_gcn_gowalla" | "light_gcn_amazon-book"
data_dict = LightGCNDataset(dataset).load_data()

num_users = data_dict["num_users"]
num_items = data_dict["num_items"]
user_item_edges = data_dict["user_item_edges"]
train_index = data_dict["train_index"]
train_user_item_edges = user_item_edges[train_index]
train_user_items_dict = data_dict["train_user_items_dict"]
test_user_items_dict = data_dict["test_user_items_dict"]

learning_rate = 5e-3
learning_rate_decay = 0.99
l2_coef = 5e-6
drop_rate = 0.2
embedding_size = 64
epochs = 3000
batch_size = 8000


class UserItemEmbedding(torch.nn.Module):

    def __init__(self, num_users, num_items, embedding_size, drop_rate=0.1):
        super().__init__()
        self.user_embeddings = torch.nn.Parameter(torch.empty([num_users, embedding_size]))
        self.item_embeddings = torch.nn.Parameter(torch.empty([num_items, embedding_size]))
        self.dropout = torch.nn.Dropout(drop_rate)

        torch.nn.init.normal_(self.user_embeddings, 0.0, 1.0 / np.sqrt(embedding_size))
        torch.nn.init.normal_(self.item_embeddings, 0.0, 1.0 / np.sqrt(embedding_size))

    def forward(self):
        user_embeddings = self.dropout(self.user_embeddings)
        item_embeddings = self.dropout(self.item_embeddings)

        return user_embeddings, item_embeddings


model = UserItemEmbedding(num_users, num_items, embedding_size, drop_rate=drop_rate).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=learning_rate_decay)
# train_data_loader = DataLoader(TensorDataset(torch.tensor(train_user_item_edges).to(device)), batch_size=batch_size, shuffle=True)


for epoch in tqdm(range(1, epochs)):

    step_losses = []
    step_mf_loss_sum = 0.0
    step_l2_losses = []

    model.train()
    for step, batch_user_item_edges in enumerate(
            tf.data.Dataset.from_tensor_slices(train_user_item_edges).shuffle(1000000).batch(batch_size)):
        # for step, (batch_user_item_edges,) in tqdm(enumerate(train_data_loader)):
        batch_user_item_edges = torch.tensor(batch_user_item_edges.numpy()).to(device)
        user_embeddings, item_embeddings = model()

        info_bpr_loss = info_bpr(user_embeddings, item_embeddings, batch_user_item_edges, num_negs=300)

        l2_loss = torch.tensor(0.0).to(device)
        for name, param in model.named_parameters():
            if "weight" in name or "embeddings" in name:
                l2_loss += 0.5 * (param ** 2).sum()

        loss = info_bpr_loss + l2_loss * l2_coef

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step_losses.append(loss.item())
        step_mf_loss_sum += info_bpr_loss * len(batch_user_item_edges)
        step_l2_losses.append(l2_loss.item())
    lr_scheduler.step()

    print("epoch = {}\tmean_loss = {}\tmean_mf_loss = {}\tmean_l2_loss = {}".format(epoch,
                                                                                    np.mean(step_losses),
                                                                                    step_mf_loss_sum / len(
                                                                                        train_user_item_edges),
                                                                                    np.mean(step_l2_losses)
                                                                                    ))
    # print("lr: ", [param_group["lr"] for param_group in optimizer.param_groups])

    if epoch % 5 == 0:
        model.eval()
        user_embeddings, item_embeddings = model()
        print("\nEvaluation before epoch {}".format(epoch))
        mean_results_dict = evaluate_mean_global_metrics(test_user_items_dict, train_user_items_dict,
                                                         user_embeddings.detach().cpu().numpy(),
                                                         item_embeddings.detach().cpu().numpy(),
                                                         k_list=[10, 20], metrics=["precision", "recall", "ndcg"])
        for metrics_name, score in mean_results_dict.items():
            print("{}: {:.4f}".format(metrics_name, score))
