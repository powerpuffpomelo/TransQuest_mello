# 根据qe信心，融合qe预测label和sim预测的label
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--conf_threshold', '-c', type=float)
parser.add_argument('--sim_threshold', '-s', type=float)
parser.add_argument('--version', '-v', type=str, default='entropy')
args = parser.parse_args()

split = 'test'
year = '19'
qe_info_prefix = '/opt/tiger/fake_arnold/TransQuest_mello/checkpoints/train_result_2021_en-de/prediction_with_confidence/'
qe_pred_path = qe_info_prefix + split + year + '.mt_tag.pred'
qe_confidence_path = qe_info_prefix + split + year + '.mt_conf.pred'

info_prefix = '/opt/tiger/fake_arnold/TransQuest_mello/checkpoints/embed_sim/'
sim_path = info_prefix + 'pred_' + args.version + '_' + split + year + '_threshold' + str(args.sim_threshold) + '.mt_tag'
save_label_prefix = '/opt/tiger/fake_arnold/TransQuest_mello/checkpoints/qe_label_augment_with_confidence/embedding_similarity/'
if not os.path.exists(save_label_prefix):
    os.makedirs(save_label_prefix, exist_ok=True)
save_label_path = save_label_prefix + args.version + '_' + split + year + '.mt_tag_conf' + str(args.conf_threshold) + '_sim' + str(args.sim_threshold) + '.pred'

with open(qe_pred_path, 'r', encoding='utf-8') as flq, open(sim_path, 'r', encoding='utf-8') as fsim, \
    open(qe_confidence_path, 'r', encoding='utf-8') as fconf, open(save_label_path, 'w', encoding='utf-8') as fs:
    qe_pred_lines = flq.readlines()
    prob_pred_lines = fsim.readlines()
    qe_conf_lines = fconf.readlines()
    for qe_pred_line, prob_pred_line, qe_conf_line in zip(qe_pred_lines, prob_pred_lines, qe_conf_lines):
        qe_pred_line = qe_pred_line.strip('\n').split()
        prob_pred_line = prob_pred_line.strip('\n').split()
        qe_conf_line = list(map(float, qe_conf_line.strip('\n').split()))
        save_label_line = [prob_pred_line[i] if qe_conf_line[i] < args.conf_threshold else qe_pred_line[i] for i in range(len(qe_pred_line))]
        fs.write(' '.join(save_label_line) + '\n')

# python3 mello_scripts/augment_by_confidence/embed_sim/3_qe_label_augment_with_sim.py -v entropy -c 0.6 -s 0.2