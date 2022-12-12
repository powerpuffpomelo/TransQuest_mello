import os

split = 'test'
year = '21'
qe_info_prefix_1 = '/opt/tiger/fake_arnold/TransQuest_mello/checkpoints/train_result_21_seed_666_en-de/prediction_with_confidence/'
qe_info_prefix_2 = '/opt/tiger/fake_arnold/TransQuest_mello/checkpoints/train_result_21_seed_777_en-de/prediction_with_confidence/'
qe_info_prefix_3 = '/opt/tiger/fake_arnold/TransQuest_mello/checkpoints/train_result_21_seed_999_en-de/prediction_with_confidence/'
pred_name = split + year + '.mt_tag.pred'
conf_name = split + year + '.mt_conf.pred'

ensemble_save_prefix = '/opt/tiger/fake_arnold/TransQuest_mello/checkpoints/ensemble/'
if not os.path.exists(ensemble_save_prefix):
    os.makedirs(ensemble_save_prefix, exist_ok=True)
ensemble_save_path = ensemble_save_prefix + split + year + '.mt_tag_ensemble_soft3.pred'

# hard
"""
with open(qe_info_prefix_1 + pred_name, 'r', encoding='utf-8') as q1, \
    open(qe_info_prefix_2 + pred_name, 'r', encoding='utf-8') as q2, \
    open(qe_info_prefix_3 + pred_name, 'r', encoding='utf-8') as q3, \
    open(ensemble_save_path, 'w', encoding='utf-8') as fs:
    q1_lines = q1.readlines()
    q2_lines = q2.readlines()
    q3_lines = q3.readlines()
    for q1_line, q2_line, q3_line in zip(q1_lines, q2_lines, q3_lines):
        q1_line = q1_line.strip('\n').split()
        q2_line = q2_line.strip('\n').split()
        q2_line = q3_line.strip('\n').split()
        save_line = []
        for w1, w2, w3 in zip(q1_line, q2_line, q3_line):
            ok_list = [1 if w == 'OK' else 0 for w in [w1, w2, w3]]
            if sum(ok_list) >= 2: save_line.append('OK')
            else: save_line.append('BAD')
        fs.write(' '.join(save_line) + '\n')
"""

# soft
with open(qe_info_prefix_1 + pred_name, 'r', encoding='utf-8') as q1, \
    open(qe_info_prefix_2 + pred_name, 'r', encoding='utf-8') as q2, \
    open(qe_info_prefix_3 + pred_name, 'r', encoding='utf-8') as q3, \
    open(qe_info_prefix_1 + conf_name, 'r', encoding='utf-8') as c1, \
    open(qe_info_prefix_2 + conf_name, 'r', encoding='utf-8') as c2, \
    open(qe_info_prefix_3 + conf_name, 'r', encoding='utf-8') as c3, \
    open(ensemble_save_path, 'w', encoding='utf-8') as fs:
    q1_lines = q1.readlines()
    q2_lines = q2.readlines()
    q3_lines = q3.readlines()
    c1_lines = c1.readlines()
    c2_lines = c2.readlines()
    c3_lines = c3.readlines()
    for q1_line, q2_line, q3_line, c1_line, c2_line, c3_line in zip(q1_lines, q2_lines, q3_lines, c1_lines, c2_lines, c3_lines):
        q1_line = q1_line.strip('\n').split()
        q2_line = q2_line.strip('\n').split()
        q3_line = q3_line.strip('\n').split()
        c1_line = c1_line.strip('\n').split()
        c2_line = c2_line.strip('\n').split()
        c3_line = c3_line.strip('\n').split()
        save_line = []
        for q1w, q2w, q3w, c1w, c2w, c3w in zip(q1_line, q2_line, q3_line, c1_line, c2_line, c3_line):
            ok_cnt = 0
            bad_cnt = 0
            for [q, c] in ([q1w, c1w], [q2w, c2w], [q3w, c3w]):
                if q == 'OK': ok_cnt += float(c)
                else: bad_cnt += float(c)
            if ok_cnt > bad_cnt: save_line.append('OK')
            else: save_line.append('BAD')
            
        fs.write(' '.join(save_line) + '\n')

# python3 /opt/tiger/fake_arnold/TransQuest_mello/mello_scripts/ensemble/get_ensemble_pred_tag.py