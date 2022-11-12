# 泛化性差的分析：19测试集中哪些token 是19训练能做对、21训练做不对的，这些token有什么特点。
import json
import numpy as np
import matplotlib.pyplot as plt

test19_gold_tag_path = '/opt/tiger/fake_arnold/wmt-qe-2019-data/test_en-de/test.mt_tag'
train19_test19_pred_tag_path = '/opt/tiger/fake_arnold/TransQuest_mello/checkpoints/train_result_2019_en-de/prediction/test2019.mt_tag.pred'
train21_test19_pred_tag_path = '/opt/tiger/fake_arnold/TransQuest_mello/checkpoints/train_result_2021_en-de/prediction/test2019.mt_tag.pred'

test19_mt_path = '/opt/tiger/fake_arnold/wmt-qe-2019-data/test_en-de/test.mt'
train21_stat_path = '/opt/tiger/fake_arnold/mlqe-pe_word_level_with_analysis/train/en-de-train/train_token_mt_tag_stat.json'
train19_stat_path = '/opt/tiger/fake_arnold/wmt-qe-2019-data/train_en-de/train_token_mt_tag_stat.json'

stat_path = '/opt/tiger/fake_arnold/TransQuest_mello/mello_scripts/analysis/result/stat_wrong.json'


def cal(test_other_gold_tag_path, train_other_test_other_pred_tag_path, train_this_test_other_pred_tag_path, \
        test_other_mt_path, train_other_stat_path, train_this_stat_path, save_path):
    info_dict = dict()
    with open(train_other_stat_path, 'r', encoding='utf-8') as fos, open(train_this_stat_path, 'r', encoding='utf-8') as fts:
        train_other_stat = json.load(fos)
        train_this_stat = json.load(fts)
    with open(test_other_gold_tag_path, 'r', encoding='utf-8') as fog, open(train_other_test_other_pred_tag_path, 'r', encoding='utf-8') as foop, \
        open(train_this_test_other_pred_tag_path, 'r', encoding='utf-8') as ftop, open(test_other_mt_path, 'r', encoding='utf-8') as fmt:
        og_lines = fog.readlines()
        oop_lines = foop.readlines()
        top_lines = ftop.readlines()
        mt_lines = fmt.readlines()

        for og_line, oop_line, top_line, mt_line in zip(og_lines, oop_lines, top_lines, mt_lines):
            og_line = og_line.strip('\n').split()
            oop_line = oop_line.strip('\n').split()
            top_line = top_line.strip('\n').split()
            mt_line = mt_line.strip('\n').split()
            for i in range(len(og_line)):
                if og_line[i] == oop_line[i] and og_line[i] != top_line[i]:   # 在其它领域测试集上，其它领域训练集能做对、本领域训练集做不对的word
                    word = mt_line[i]
                    if word not in info_dict:
                        info_dict[word] = {'gold_label_list':[og_line[i]], 'pred_wrong_freq':1, \
                                'other_train_freq':0, 'this_train_freq':0, \
                                'other_train_ok_ratio':-1, 'this_train_ok_ratio':-1}
                        if word in train_other_stat: 
                            info_dict[word]['other_train_freq'] = train_other_stat[word]['train_freq']
                            info_dict[word]['other_train_ok_ratio'] = train_other_stat[word]['ok_ratio']
                        if word in train_this_stat:
                            info_dict[word]['this_train_freq'] = train_this_stat[word]['train_freq']
                            info_dict[word]['this_train_ok_ratio'] = train_this_stat[word]['ok_ratio']
                    else:
                        info_dict[word]['pred_wrong_freq'] += 1
                        info_dict[word]['gold_label_list'].append(og_line[i])

    with open(save_path, 'w', encoding='utf-8') as fs:
        json.dump(info_dict, fs, indent=4)

# cal(test_other_gold_tag_path=test19_gold_tag_path,\
#     train_other_test_other_pred_tag_path=train19_test19_pred_tag_path,\
#     train_this_test_other_pred_tag_path=train21_test19_pred_tag_path,\
#     test_other_mt_path=test19_mt_path,\
#     train_other_stat_path=train19_stat_path,\
#     train_this_stat_path=train21_stat_path,\
#     save_path=stat_path)

def plot_freq(info_path, save_path):
    with open(info_path, 'r', encoding='utf-8') as f:
        info_dict = json.load(f)
    other_train_freq_list = [v['other_train_freq'] for v in info_dict.values()]
    this_train_freq_list = [v['this_train_freq'] for v in info_dict.values()]
    zot = list(zip(other_train_freq_list, this_train_freq_list))
    zot.sort(key = lambda x : (x[0], x[1]), reverse=True)

    list1, list2 = zip(*zot)
    #print(sum(list1) / len(list1))
    #print(sum(list2) / len(list2))
    x = np.arange(len(list1))
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.fill_between(x, list1, y2=0, label='2019 Train Freq',  facecolor='tomato', alpha=0.6)
    ax.fill_between(x, list2, y2=0, label='2021 Train Freq', color='deepskyblue', alpha=0.8)
    ax.set_xlim((0, 200))
    ax.set_ylim((0, 2000))
    #ax.set_xlabel('every wrong pred word', fontsize=24)
    ax.set_ylabel('Train Freq', fontsize=16)
    ax.set_title("2019 Wrong Pred Words", fontsize=18)
    ax.legend(fontsize=16)
    plt.savefig(save_path)

#plot_path = '/opt/tiger/fake_arnold/TransQuest_mello/mello_scripts/analysis/result/plot_freq.jpg'
#plot_freq(info_path=stat_path, save_path=plot_path)

def plot_dist(info_path, save_path):
    with open(info_path, 'r', encoding='utf-8') as f:
        info_dict = json.load(f)
    
    gold_label_list = []
    other_train_ok_ratio_list = []
    this_train_ok_ratio_list = []
    for k,v in info_dict.items():
        temp_gold_list = v['gold_label_list']
        for tag in temp_gold_list:
            if tag == 'OK':
                gold_label_list.append(1)
            else:
                gold_label_list.append(0)
            other_train_ok_ratio_list.append(v['other_train_ok_ratio']) 
            this_train_ok_ratio_list.append(v['this_train_ok_ratio'])
    
    zot = list(zip(other_train_ok_ratio_list, this_train_ok_ratio_list, gold_label_list))
    zot = [x for x in zot if x[0] != -1 and x[1] != -1]
    vec_ok = [[x[0], x[1]] for x in zot if x[2] == 1]
    vec_bad = [[x[0], x[1]] for x in zot if x[2] == 0]

    vec_ok.sort(key=lambda x : (x[0], x[1]), reverse=True)
    vec_bad.sort(key=lambda x : (x[0], x[1]), reverse=True)

    gold_ok_other, gold_ok_this = map(list, zip(*vec_bad))
    gold_ok_this.sort(reverse=True)
    x = np.arange(len(gold_ok_other))
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.fill_between(x, gold_ok_other, y2=0, label='19_train_ok_ratio',  facecolor='tomato', alpha=0.6)
    ax.fill_between(x, gold_ok_this, y2=0, label='21_train_ok_ratio', color='deepskyblue', alpha=0.8)
    ax.set_xlim((0, len(gold_ok_other) - 1))
    ax.set_ylim((0, 1))
    ax.set_ylabel('train ok ratio', fontsize=16)
    ax.set_title("2019 Wrong Pred Words when gold=bad", fontsize=18)
    ax.legend(fontsize=16)
    plt.savefig(save_path)

    # ============ 混合gold label ============ #
    # vec = [[abs(x[0] - x[2]), abs(x[1] - x[2])] for x in zot]
    # vec.sort(key=lambda x:(x[1], x[0]), reverse=True)
    # dist_other_list, dist_this_list = map(list, zip(*vec))

    # x = np.arange(len(dist_other_list))
    # fig, ax = plt.subplots(figsize=(12, 8))
    # ax.fill_between(x, dist_this_list, y2=0, label='dist to 2021 train ok ratio',  facecolor='tomato', alpha=0.6)
    # ax.fill_between(x, dist_other_list, y2=0, label='dist to 2019 train ok ratio', color='deepskyblue', alpha=0.8)
    # ax.set_xlim((0, 500))
    # ax.set_ylim((0, 1))
    # ax.set_ylabel('dist of train ok ratio', fontsize=16)
    # ax.set_title("2019 Wrong Pred Words", fontsize=18)
    # ax.legend(fontsize=16)
    # plt.savefig(save_path)

plot_path = '/opt/tiger/fake_arnold/TransQuest_mello/mello_scripts/analysis/result/plot_dist.jpg'
plot_dist(info_path=stat_path, save_path=plot_path)

def plot_freq_dist(info_path, save_path):
    with open(info_path, 'r', encoding='utf-8') as f:
        info_dict = json.load(f)
    gold_label_list = [1 if v['gold_label'] == 'OK' else 0 for v in info_dict.values()]
    other_train_freq_list = [v['this_train_freq'] for v in info_dict.values()]
    other_train_ok_ratio_list = [v['this_train_ok_ratio'] for v in info_dict.values()]
    zot = list(zip(other_train_freq_list, other_train_ok_ratio_list, gold_label_list))
    zot = [x for x in zot if x[0] != 0]
    vec = [[x[0], abs(x[1] - x[2])] for x in zot]
    vec.sort(key=lambda x:(x[0], x[1]), reverse=True)
    freq_other_list, dist_other_list = map(list, zip(*vec))
    x = np.arange(len(dist_other_list))
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot()
    ax2 = ax1.twinx()

    ax1.fill_between(x, freq_other_list, y2=0, label='2021 train freq',  facecolor='tomato', alpha=0.6)
    ax2.fill_between(x, dist_other_list, y2=0, label='dist to 2021 train ok ratio', color='deepskyblue', alpha=0.8)
    # ax.set_xlim((0, 230))
    # ax.set_ylim((0, 1))
    ax1.set_ylabel('2021 train freq', fontsize=16)
    ax2.set_ylabel('distribution diff', fontsize=16)
    ax1.set_title("2019 Wrong Pred Words", fontsize=18)
    fig.legend(fontsize=16)
    plt.savefig(save_path)


# plot_path = '/opt/tiger/fake_arnold/TransQuest_mello/mello_scripts/analysis/result/plot_freq_dist.jpg'
# plot_freq_dist(info_path=stat_path, save_path=plot_path)

# python3 /opt/tiger/fake_arnold/TransQuest_mello/mello_scripts/analysis/stat/stat_token_pred_wrong.py