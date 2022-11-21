# 通过阈值，把force decoding概率转化成qe label
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--threshold', '-t', type=float)
args = parser.parse_args()

split = 'test'
year = '19'
trans_info_prefix = '/opt/tiger/fake_arnold/TransQuest_mello/checkpoints/translation_prob/t5_large/'
force_decoding_prob_path = trans_info_prefix + 'prob_' + split + year + '.txt'
save_label_path = trans_info_prefix + 'pred_' + split + year + '_threshold' + str(args.threshold) + '.mt_tag'

with open(force_decoding_prob_path, 'r', encoding='utf-8') as fprob, \
        open(save_label_path, 'w', encoding='utf-8') as fs:
    prob_lines = fprob.readlines()
    for prob_line in prob_lines:
        prob_line = list(map(float, prob_line.strip('\n').split()))
        label_line = ['OK' if p >= args.threshold else 'BAD' for p in prob_line]
        fs.write(' '.join(label_line) + '\n')

# python3 mello_scripts/augment_by_confidence/force_decoding_prob_2_qe_label_by_threshold.py -t 0.2