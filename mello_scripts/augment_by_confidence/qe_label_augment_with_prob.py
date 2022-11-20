# 根据qe信心，融合qe预测label和force decoding概率预测的label
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--conf_threshold', '-c', type=float)
parser.add_argument('--prob_threshold', '-p', type=float)
args = parser.parse_args()

split = 'test'
year = '2021'
qe_pred_path = '/opt/tiger/fake_arnold/TransQuest_mello/checkpoints/train_result_2021_en-de/prediction_with_confidence/' + split + year + '.mt_tag.pred'
qe_confidence_path = '/opt/tiger/fake_arnold/TransQuest_mello/checkpoints/train_result_2021_en-de/prediction_with_confidence/' + split + year + '.mt_conf.pred'
force_decoding_prob_path = '/opt/tiger/fake_arnold/TransQuest_mello/checkpoints/translation_prob/t5_large_pred_' + split + year + '_threshold' + str(args.prob_threshold) + '.mt_tag'

save_label_path = '/opt/tiger/fake_arnold/TransQuest_mello/checkpoints/qe_label_augment_with_confidence/' + split + year + '.mt_tag_conf' + str(args.conf_threshold) + '_prob' + str(args.prob_threshold) + '.pred'

with open(qe_pred_path, 'r', encoding='utf-8') as flq, open(force_decoding_prob_path, 'r', encoding='utf-8') as flp, \
    open(qe_confidence_path, 'r', encoding='utf-8') as fconf, open(save_label_path, 'w', encoding='utf-8') as fs:
    qe_pred_lines = flq.readlines()
    prob_pred_lines = flp.readlines()
    qe_conf_lines = fconf.readlines()
    for qe_pred_line, prob_pred_line, qe_conf_line in zip(qe_pred_lines, prob_pred_lines, qe_conf_lines):
        qe_pred_line = qe_pred_line.strip('\n').split()
        prob_pred_line = prob_pred_line.strip('\n').split()
        qe_conf_line = list(map(float, qe_conf_line.strip('\n').split()))
        save_label_line = [prob_pred_line[i] if qe_conf_line[i] < args.conf_threshold and prob_pred_line[i] != -1 else qe_pred_line[i] for i in range(len(qe_pred_line))]
        fs.write(' '.join(save_label_line) + '\n')

# python3 mello_scripts/augment_by_confidence/qe_label_augment_with_prob.py -c 0.6 -p 0.2