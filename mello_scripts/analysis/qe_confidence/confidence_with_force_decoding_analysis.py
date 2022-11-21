# 分析，引入翻译概率增强qe后，提升的部分在哪、之前的低频词有多少改对了
import json
import numpy as np
import matplotlib.pyplot as plt

test19_gold_tag_path = '/opt/tiger/fake_arnold/qe_data/qe_data_mello/test19/en-de-test19/test19.mt_tag'
train19_test19_pred_tag_path = '/opt/tiger/fake_arnold/TransQuest_mello/checkpoints/train_result_2019_en-de/prediction/test2019.mt_tag.pred'
train21_test19_pred_tag_path = '/opt/tiger/fake_arnold/TransQuest_mello/checkpoints/train_result_2021_en-de/prediction_with_confidence/test19.mt_tag.pred'
prob_aug_test19_pred_tag_path = '/opt/tiger/fake_arnold/TransQuest_mello/checkpoints/qe_label_augment_with_confidence/test19.mt_tag_conf0.6_prob0.2.pred'

test19_mt_path = '/opt/tiger/fake_arnold/qe_data/qe_data_mello/test19/en-de-test19/test19.mt'

conf_threshold = 0.8
qe_conf_path = '/opt/tiger/fake_arnold/TransQuest_mello/checkpoints/train_result_2021_en-de/prediction_with_confidence/test19.mt_conf.pred'

def cal(test_other_gold_tag_path, test_other_mt_path, \
        train_other_test_other_pred_tag_path, train_this_test_other_pred_tag_path, \
        aug_test_other_pred_tag_path):

    with open(test_other_gold_tag_path, 'r', encoding='utf-8') as fog, open(train_other_test_other_pred_tag_path, 'r', encoding='utf-8') as foop, \
        open(train_this_test_other_pred_tag_path, 'r', encoding='utf-8') as ftop, \
        open(aug_test_other_pred_tag_path, 'r', encoding='utf-8') as fap, \
        open(test_other_mt_path, 'r', encoding='utf-8') as fmt, \
        open(qe_conf_path, 'r', encoding='utf-8') as fconf:
        og_lines = fog.readlines()
        oop_lines = foop.readlines()
        top_lines = ftop.readlines()
        ap_lines = fap.readlines()
        mt_lines = fmt.readlines()
        conf_lines = fconf.readlines()

        cnt_all_word = 0
        cnt_low_conf = 0   # 信心低于阈值
        
        cnt_qe_right = 0
        cnt_trans_right = 0
        cnt_all_right = 0
        cnt_all_wrong = 0
        # cnt_gen_wrong = 0   # 错误类型：泛化性错误：在其它领域测试集上，其它领域训练集能做对、本领域训练集做不对的word
        # cnt_gen_wrong_change_right = 0   # 又被翻译概率改对的
        cnt_gen_wrong_low_conf = 0
        cnt_not_gen_wrong_low_conf = 0
        cnt_gen_wrong_not_low_conf = 0
        cnt_not_not = 0


        for og_line, oop_line, top_line, ap_line, mt_line, conf_line in zip(og_lines, oop_lines, top_lines, ap_lines, mt_lines, conf_lines):
            og_line = og_line.strip('\n').split()
            oop_line = oop_line.strip('\n').split()
            top_line = top_line.strip('\n').split()
            ap_line = ap_line.strip('\n').split()
            mt_line = mt_line.strip('\n').split()
            conf_line = list(map(float, conf_line.strip('\n').split()))
            for i in range(len(og_line)):
                cnt_all_word += 1
                # if og_line[i] == oop_line[i] and og_line[i] != top_line[i]:   # 在其它领域测试集上，其它领域训练集能做对、本领域训练集做不对的word
                #     cnt_gen_wrong += 1
                #     if conf_line[i] < conf_threshold: cnt_low_conf += 1
                    # if og_line[i] == ap_line[i]: cnt_gen_wrong_change_right += 1

                if (og_line[i] == oop_line[i] and og_line[i] != top_line[i]) and conf_line[i] < conf_threshold:
                    cnt_gen_wrong_low_conf += 1
                elif not (og_line[i] == oop_line[i] and og_line[i] != top_line[i]) and conf_line[i] < conf_threshold:
                    cnt_not_gen_wrong_low_conf += 1
                elif (og_line[i] == oop_line[i] and og_line[i] != top_line[i]) and not conf_line[i] < conf_threshold:
                    cnt_gen_wrong_not_low_conf += 1
                elif not (og_line[i] == oop_line[i] and og_line[i] != top_line[i]) and not conf_line[i] < conf_threshold:
                    cnt_not_not += 1
                #     if top_line[i] == og_line[i] and ap_line[i] == og_line[i]: cnt_all_right += 1
                #     elif top_line[i] == og_line[i] and ap_line[i] != og_line[i]: cnt_qe_right += 1
                #     elif top_line[i] != og_line[i] and ap_line[i] == og_line[i]: cnt_trans_right += 1
                #     elif top_line[i] != og_line[i] and ap_line[i] != og_line[i]: cnt_all_wrong += 1

    # print(cnt_low_conf / cnt_all_word)
    # print(cnt_all_right / cnt_low_conf)
    # print(cnt_qe_right / cnt_low_conf)
    # print(cnt_trans_right / cnt_low_conf)
    # print(cnt_all_wrong / cnt_low_conf)

    # print(cnt_gen_wrong / cnt_low_conf)
    print(cnt_gen_wrong_low_conf / cnt_all_word)
    print(cnt_not_gen_wrong_low_conf / cnt_all_word)
    print(cnt_gen_wrong_not_low_conf / cnt_all_word)
    print(cnt_not_not / cnt_all_word)

    p = cnt_gen_wrong_low_conf / (cnt_gen_wrong_low_conf + cnt_not_gen_wrong_low_conf)
    r = cnt_gen_wrong_low_conf / (cnt_gen_wrong_low_conf + cnt_gen_wrong_not_low_conf)
    print(p)
    print(r)
    print(2 * p * r / (p + r))

cal(test_other_gold_tag_path=test19_gold_tag_path,\
    train_other_test_other_pred_tag_path=train19_test19_pred_tag_path,\
    train_this_test_other_pred_tag_path=train21_test19_pred_tag_path,\
    aug_test_other_pred_tag_path=prob_aug_test19_pred_tag_path,\
    test_other_mt_path=test19_mt_path)

# python3 mello_scripts/analysis/qe_confidence/confidence_with_force_decoding_analysis.py