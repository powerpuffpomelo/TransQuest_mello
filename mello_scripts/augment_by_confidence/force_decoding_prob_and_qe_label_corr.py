# 先导实验，计算force decoding概率 和 qe标签 的相关性，看是否能用force decoding来做qe任务
from scipy.stats import pearsonr

force_decoding_prob_path = '/opt/tiger/fake_arnold/TransQuest_mello/checkpoints/translation_prob/t5_prob.txt'
qe_gold_label_path = '/opt/tiger/fake_arnold/qe_data/wmt-qe-2019-data/test_en-de/test.mt_tag'

prob_list = []
label_list = []
with open(force_decoding_prob_path, 'r', encoding='utf-8') as fprob, \
        open(qe_gold_label_path, 'r', encoding='utf-8') as flabel:
    prob_lines = fprob.readlines()
    label_lines = flabel.readlines()
    for prob_line, label_line in zip(prob_lines, label_lines):
        prob_line = map(float, prob_line.strip('\n').split())
        label_line = label_line.strip('\n').split()
        label_line = [1 if l == 'OK' else 0 for l in label_line]
        prob_list.extend(prob_line)
        label_list.extend(label_line)
    
print(pearsonr(prob_list, label_list))

# python3 mello_scripts/augment_by_confidence/force_decoding_prob_and_qe_label_corr.py