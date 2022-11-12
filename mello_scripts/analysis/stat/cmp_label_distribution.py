# 统计 两个集合之间的 标签分布差异
import json
from scipy.stats import pearsonr
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

train21_path = '/opt/tiger/fake_arnold/mlqe-pe_word_level_with_analysis/train/en-de-train/train_token_mt_tag_stat.json'
train19_path = '/opt/tiger/fake_arnold/wmt-qe-2019-data/train_en-de/train_token_mt_tag_stat.json'
test21_path = '/opt/tiger/fake_arnold/mlqe-pe_word_level_with_analysis/test21/en-de-test21/test2021.token_mt_tag_stat.json'
test19_path = '/opt/tiger/fake_arnold/wmt-qe-2019-data/test_en-de/test_token_mt_tag_stat.json'

def get_two_ok_ratio_list(info1_path, info2_path):
    with open(info1_path, 'r', encoding='utf-8') as f1, open(info2_path, 'r', encoding='utf-8') as f2:
        info1 = json.load(f1)
        info2 = json.load(f2)

    ok_ratio_list1 = []
    ok_ratio_list2 = []

    for k in info1.keys():
        if k in info2.keys():
            ok_ratio_list1.append(info1[k]['ok_ratio'])
            ok_ratio_list2.append(info2[k]['ok_ratio'])
        
    assert len(ok_ratio_list1) == len(ok_ratio_list2)

    ok_ratio = list(zip(ok_ratio_list1, ok_ratio_list2))
    ok_ratio.sort(key=lambda x : (x[0], x[1]), reverse = True)
    ok_ratio_list1, ok_ratio_list2 = zip(*ok_ratio)
    return ok_ratio_list1, ok_ratio_list2



def cal_pearsonr(ok_ratio_list1, ok_ratio_list2):
    print(pearsonr(ok_ratio_list1, ok_ratio_list2))   # r,p 相关系数，显著性

    """
    (0.21583914801341678, 5.2161555846217386e-104) # 17train, 19train   17 19训练集有重合
    (0.05629978492979142, 0.0016075356703827114)   # 17train, 21train
    (0.027734499648744873, 0.16899768475014446)    # 19train, 21train

    (0.25442873680296035, 8.277414377837152e-64)   # 17train, 17test
    (0.19945987432094733, 1.7203760954702344e-26)  # 19train, 19test
    (0.1851500989257273, 3.171795589169726e-27)    # 21train, 21test
    """

def plot(lists, labels, save_path):
    list1, list2 = lists
    x = np.arange(len(list1))
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.scatter(x, list1, label=labels[0], marker='x', color='tomato', alpha=1)
    #ax.scatter(x, list2, label=labels[1], marker='x', color='deepskyblue', alpha=0.4)
    ax.scatter(x, list2, label=labels[1], marker='x', color='aquamarine', alpha=1)
    ax.set_xlabel('Every Word', fontsize=24)
    ax.set_ylabel('OK Ratio', fontsize=24)
    ax.set_title("Compare Label Distribution: 21train & 21test", fontsize=28)
    ax.legend(fontsize=24)
    plt.savefig(save_path)

# 19train, 21train
# ok_ratio_list1, ok_ratio_list2 = get_two_ok_ratio_list(train19_path, train21_path)
# lists = [ok_ratio_list1, ok_ratio_list2]
# labels = ['19train', '21train']
# save_path = '/opt/tiger/fake_arnold/TransQuest_mello/mello_scripts/analysis/result/cmp_19_21_train.jpg'

# 19train, 19test
# ok_ratio_list1, ok_ratio_list2 = get_two_ok_ratio_list(train19_path, test19_path)
# lists = [ok_ratio_list1, ok_ratio_list2]
# labels = ['19train', '19test']
# save_path = '/opt/tiger/fake_arnold/TransQuest_mello/mello_scripts/analysis/result/cmp_19_train_test.jpg'

# 21train, 19train
# ok_ratio_list1, ok_ratio_list2 = get_two_ok_ratio_list(train21_path, train19_path)
# lists = [ok_ratio_list1, ok_ratio_list2]
# labels = ['21train', '19train']
# save_path = '/opt/tiger/fake_arnold/TransQuest_mello/mello_scripts/analysis/result/cmp_21_19_train.jpg'

# 21train, 21test
ok_ratio_list1, ok_ratio_list2 = get_two_ok_ratio_list(train21_path, test21_path)
lists = [ok_ratio_list1, ok_ratio_list2]
labels = ['21train', '21test']
save_path = '/opt/tiger/fake_arnold/TransQuest_mello/mello_scripts/analysis/result/cmp_21_train_test.jpg'

plot(lists, labels, save_path)

# python3 mello_scripts/analysis/stat/cmp_label_distribution.py