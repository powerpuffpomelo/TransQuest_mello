# qe模型的预测信心，和训练词频相关吗？和预测准确率相关吗？
import json
from scipy.stats import pearsonr

pred_data_prefix = '/opt/tiger/fake_arnold/qe_data/wmt-qe-2019-data/test_en-de/'
pred_mt_path = pred_data_prefix + 'test.mt'
pred_gold_tag_path = pred_data_prefix + 'test.mt_tag'
pred_info_path = pred_data_prefix + 'test_token_mt_tag_stat.json'
pred_pred_tag_path = '/opt/tiger/fake_arnold/TransQuest_mello/checkpoints/train_result_2021_en-de/prediction_with_confidence/test2019.mt_tag.pred'
pred_conf_path = '/opt/tiger/fake_arnold/TransQuest_mello/checkpoints/train_result_2021_en-de/prediction_with_confidence/test2019.mt_conf.pred'

train_data_prefix = '/opt/tiger/fake_arnold/qe_data/mlqe-pe_word_level_with_analysis/train/en-de-train/'
train_info_path = train_data_prefix + 'train_token_mt_tag_stat.json'

with open(pred_info_path, 'r', encoding='utf-8') as fpi, open(train_info_path, 'r', encoding='utf-8') as fti:
    pred_info = json.load(fpi)
    train_info = json.load(fti)

train_freq_list = []
pred_conf_list = []
pred_right_list = []
with open(pred_mt_path, 'r', encoding='utf-8') as fmt, open(pred_conf_path, 'r', encoding='utf-8') as fc, \
    open(pred_gold_tag_path, 'r', encoding='utf-8') as fg, open(pred_pred_tag_path, 'r', encoding='utf-8') as fp:
    pred_mt_lines = fmt.readlines()
    pred_conf_lines = fc.readlines()
    pred_gold_tag_lines = fg.readlines()
    pred_pred_tag_lines = fp.readlines()
    for mt_line, conf_line, gtag_line, ptag_line in zip(pred_mt_lines, pred_conf_lines, pred_gold_tag_lines, pred_pred_tag_lines):
        mt_line = mt_line.strip('\n').split()
        conf_line = conf_line.strip('\n').split()
        conf_line = list(map(float, conf_line))
        gtag_line = gtag_line.strip('\n').split()
        ptag_line = ptag_line.strip('\n').split()
        for w, c, g, p in zip(mt_line, conf_line, gtag_line, ptag_line):
            if w in train_info:
                train_freq_list.append(train_info[w]['train_freq'])
            else:
                train_freq_list.append(0)
            pred_conf_list.append(c)
            if g == p:
                pred_right_list.append(1) # 预测正确
            else:
                pred_right_list.append(0) # 预测错误

print(pearsonr(pred_conf_list, train_freq_list))
print(pearsonr(pred_conf_list, pred_right_list))
print(pearsonr(train_freq_list, pred_right_list))
"""
(0.39849954619983297, 0.0)
(0.25726512170327687, 6.5086943213715745e-285)
(0.11108534250453052, 3.044374946835908e-53)
"""

# python3 mello_scripts/analysis/qe_confidence/confidence_train_freq_corr.py