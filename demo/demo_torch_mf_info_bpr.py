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
from torch.utils.data import Dataset, DataLoader, TensorDataset

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


learning_rate = 1e-2
l2_coef = 1e-5
drop_rate = 0.1
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


model = UserItemEmbedding(num_users, num_items, embedding_size).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# train_data_loader = DataLoader(TensorDataset(torch.tensor(train_user_item_edges).to(device)), batch_size=batch_size, shuffle=True)


for epoch in tqdm(range(1, epochs)):

    step_losses = []
    step_mf_loss_sum = 0.0
    step_l2_losses = []

    model.train()
    for step, batch_user_item_edges in enumerate(tf.data.Dataset.from_tensor_slices(train_user_item_edges).shuffle(1000000).batch(batch_size)):
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


    print("epoch = {}\tmean_loss = {}\tmean_mf_loss = {}\tmean_l2_loss = {}".format(epoch,
                                                                                   np.mean(step_losses),
                                                                                   step_mf_loss_sum / len(train_user_item_edges),
                                                                                   np.mean(step_l2_losses)
                                                                                   ))

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

asdfsad





#
# for _ in range(epochs):
#
#
#
#
# def train_step(batch_user_indices, batch_item_indices, batch_neg_item_indices):
#     with tf.GradientTape() as tape:
#         embedded_users, [embedded_items, embedded_neg_items] = \
#             embedding_model([batch_user_indices, [batch_item_indices, batch_neg_item_indices]], training=True)
#
#         pos_logits = tf.reduce_sum(embedded_users * embedded_items, axis=-1)
#         neg_logits = tf.reduce_sum(embedded_users * embedded_neg_items, axis=-1)
#         #
#         # pos_losses = tf.nn.sigmoid_cross_entropy_with_logits(
#         #     logits=pos_logits,
#         #     labels=tf.ones_like(pos_logits)
#         # )
#         # neg_losses = tf.nn.sigmoid_cross_entropy_with_logits(
#         #     logits=neg_logits,
#         #     labels=tf.zeros_like(neg_logits)
#         # )
#         #
#         # losses = pos_losses + neg_losses
#
#         mf_losses = tf.nn.softplus(-(pos_logits - neg_logits))
#
#         l2_vars = [var for var in tape.watched_variables() if "kernel" in var.name or "embeddings" in var.name]
#         l2_losses = [tf.nn.l2_loss(var) for var in l2_vars]
#         l2_loss = tf.add_n(l2_losses)
#
#         loss = tf.reduce_sum(mf_losses) + l2_loss * l2
#
#     vars = tape.watched_variables()
#     grads = tape.gradient(loss, vars)
#     optimizer.apply_gradients(zip(grads, vars))
#
#     return loss, mf_losses, l2_loss
#
#
# for epoch in range(0, epoches):
#
#     if epoch % 20 == 0:
#         print("\nEvaluation before epoch {}".format(epoch))
#         mean_results_dict = evaluate_mean_global_metrics(test_user_items_dict, train_user_items_dict,
#                                                          embedding_model.user_embeddings,
#                                                          embedding_model.item_embeddings,
#                                                          k_list=[10, 20], metrics=["precision", "recall", "ndcg"])
#         for metrics_name, score in mean_results_dict.items():
#             print("{}: {:.4f}".format(metrics_name, score))
#         print()
#
#     step_losses = []
#     step_mf_losses_list = []
#     step_l2_losses = []
#
#     start_time = time()
#
#     for step, batch_edges in enumerate(
#             tf.data.Dataset.from_tensor_slices(train_user_item_edges).shuffle(1000000).batch(batch_size)):
#         batch_user_indices = batch_edges[:, 0]
#         batch_item_indices = batch_edges[:, 1]
#         batch_neg_item_indices = np.random.randint(0, num_items, batch_item_indices.shape)
#
#         loss, mf_losses, l2_loss = train_step(batch_user_indices, batch_item_indices, batch_neg_item_indices)
#
#         step_losses.append(loss.numpy())
#         step_mf_losses_list.append(mf_losses.numpy())
#         step_l2_losses.append(l2_loss.numpy())
#
#     end_time = time()
#
#     if optimizer.learning_rate.numpy() > 1e-6:
#         optimizer.learning_rate.assign(optimizer.learning_rate * 0.995)
#         lr_status = "update lr => {:.4f}".format(optimizer.learning_rate.numpy())
#     else:
#         lr_status = "current lr => {:.4f}".format(optimizer.learning_rate.numpy())
#
#     print("epoch = {}\tloss = {:.4f}\tmf_loss = {:.4f}\tl2_loss = {:.4f}\t{}\tepoch_time = {:.4f}s".format(
#         epoch, np.mean(step_losses), np.mean(np.concatenate(step_mf_losses_list, axis=0)),
#         np.mean(step_l2_losses), lr_status, end_time - start_time))
#
#     if epoch == 1:
#         print("the first epoch may take a long time to compile tf.function")