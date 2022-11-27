# 先导实验，计算词向量相似度 和 qe标签 的相关性，看是否能用词向量相似度来做qe任务
from scipy.stats import pearsonr

embedding_similarity_path = '/opt/tiger/fake_arnold/TransQuest_mello/checkpoints/embed_sim/sim_entropy_test19.txt'
qe_gold_label_path = '/opt/tiger/fake_arnold/qe_data/qe_data_mello/test19/en-de-test19/test19.mt_tag'

sim_list = []
label_list = []
with open(embedding_similarity_path, 'r', encoding='utf-8') as fsim, \
        open(qe_gold_label_path, 'r', encoding='utf-8') as flabel:
    sim_lines = fsim.readlines()
    label_lines = flabel.readlines()
    for sim_line, label_line in zip(sim_lines, label_lines):
        sim_line = map(float, sim_line.strip('\n').split())
        label_line = label_line.strip('\n').split()
        label_line = [1 if l == 'OK' else 0 for l in label_line]
        sim_list.extend(sim_line)
        label_list.extend(label_line)

assert len(sim_list) == len(label_list)

print(pearsonr(sim_list, label_list))

# python3 mello_scripts/augment_by_confidence/embed_sim/analysis/embed_sim_and_qe_label_corr.py