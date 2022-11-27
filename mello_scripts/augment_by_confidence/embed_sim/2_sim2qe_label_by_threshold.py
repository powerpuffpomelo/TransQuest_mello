# 通过阈值，把sim转化成qe label
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--sim_threshold', '-s', type=float)
parser.add_argument('--version', '-v', type=str, default='entropy')
args = parser.parse_args()

split = 'test'
year = '21'
info_prefix = '/opt/tiger/fake_arnold/TransQuest_mello/checkpoints/embed_sim/'
info_path = info_prefix + 'sim_' + args.version + '_' + split + year + '.txt'
save_label_path = info_prefix + 'pred_' + args.version + '_' + split + year + '_threshold' + str(args.sim_threshold) + '.mt_tag'

with open(info_path, 'r', encoding='utf-8') as fprob, \
        open(save_label_path, 'w', encoding='utf-8') as fs:
    prob_lines = fprob.readlines()
    for prob_line in prob_lines:
        prob_line = list(map(float, prob_line.strip('\n').split()))
        if args.version == 'max':
            label_line = ['OK' if p >= args.sim_threshold else 'BAD' for p in prob_line]
        elif args.version == 'entropy':
            label_line = ['OK' if p <= args.sim_threshold else 'BAD' for p in prob_line]  # nan也是bad
        fs.write(' '.join(label_line) + '\n')

# python3 mello_scripts/augment_by_confidence/embed_sim/2_sim2qe_label_by_threshold.py -v entropy -s 0.4