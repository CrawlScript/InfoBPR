# coding=utf-8

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import math
import os
import seaborn as sns



sns.set_theme(style="ticks")


SMALL_SIZE = 15
MEDIUM_SIZE = 15
BIGGER_SIZE = 15

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('legend', title_fontsize=BIGGER_SIZE)


# sns.set_context("paper", rc={"font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10})

csv_base_dir = "csv_results"

# csv_name = "bpr_yelp"
csv_name = "Yelp2018"

csv_fpath = os.path.join(csv_base_dir, "{}.csv".format(csv_name))


df = pd.read_csv(csv_fpath)
max_score = df["NDCG@20"].max()
base_score = df["NDCG@20"].min()



# fig, ax = plt.subplots(figsize=(13, 8))
fig, ax = plt.subplots(figsize=(11, 9))

ax.xaxis.labelpad = 10


ax.yaxis.set_label_position("right")

g = sns.barplot(
    ax=ax,
    data=df,
    x="Method", y="NDCG@20",
    palette="deep", alpha=1.0
)
sns.despine(fig, ax)
g.set(ylim=(base_score - 0.01, max_score + 0.005))
g.axhline(base_score)
g.set(xlabel="Method")


plt.show()
# plt.savefig("plots/{}.png".format(csv_name), bbox_inches='tight')
