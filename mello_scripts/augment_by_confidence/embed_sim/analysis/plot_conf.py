from matplotlib import pyplot as plt
import numpy as np

split = 'test'
year = '19'
qe_info_prefix = '/opt/tiger/fake_arnold/TransQuest_mello/checkpoints/train_result_2021_en-de/prediction_with_confidence/'
qe_confidence_path = qe_info_prefix + split + year + '.mt_conf.pred'
qe_gold_path = '/opt/tiger/fake_arnold/qe_data/qe_data_mello/' + split + year + '/en-de-' + split + year + '/' + split + year + '.mt_tag'
qe_pred_path = '/opt/tiger/fake_arnold/TransQuest_mello/checkpoints/train_result_2021_en-de/prediction_with_confidence/' + split + year + '.mt_tag.pred'


def plot_conf_label():
    plot_path = '/opt/tiger/fake_arnold/TransQuest_mello/mello_scripts/augment_by_confidence/embed_sim/analysis/plot_conf_label.jpg'

    N = 5
    ok_stat = [0] * N
    bad_stat = [0] * N

    with open(qe_confidence_path, 'r', encoding='utf-8') as fc, \
        open(qe_gold_path, 'r', encoding='utf-8') as fg:
        conf_lines = fc.readlines()
        gold_lines = fg.readlines()

        for conf_line, gold_line in zip(conf_lines, gold_lines):
            conf_line = list(map(float, conf_line.strip('\n').split()))
            gold_line = gold_line.strip('\n').split()
            for c, g in zip(conf_line, gold_line):
                if c >= 0.5 and c <= 0.6: i = 0
                elif c > 0.6 and c <= 0.7: i = 1
                elif c > 0.7 and c <= 0.8: i = 2
                elif c > 0.8 and c <= 0.9: i = 3
                elif c > 0.9 and c <= 1.0: i = 4
                if g == 'OK': ok_stat[i] += 1
                elif g == 'BAD': bad_stat[i] += 1

    print(ok_stat)
    print(bad_stat)
    bad_ratio = [bad / (ok + bad) for ok, bad in zip(ok_stat, bad_stat)]

    x = np.arange(N)
    width = 0.35

    fig, ax = plt.subplots()

    p1 = ax.bar(x, ok_stat, width, label='OK', color='r', alpha=0.6)
    p2 = ax.bar(x, bad_stat, width, bottom=ok_stat, 
                    label='BAD', color='blue', alpha=0.3)

    ax.plot(x, bad_ratio)
    # ax.axhline(0, color='grey', linewidth=0.8)
    ax.set_xlabel('QE confidence')
    ax.set_ylabel('num of OK & BAD')
    ax.set_title('num of OK & BAD group by QE confidence')
    ax.set_xticks(x, labels=['0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0'])
    ax.legend()

    # Label with label_type 'center' instead of the default 'edge'
    ax.bar_label(p1, label_type='center')
    ax.bar_label(p2, label_type='center')
    # ax.bar_label(bad_ratio)
    for i in range(N):
            plt.text(x[i], ok_stat[i] + bad_stat[i] + 100, str(round(bad_ratio[i], 2)), ha='center')

    plt.savefig(plot_path)

def plot_conf_acc():
    plot_path = '/opt/tiger/fake_arnold/TransQuest_mello/mello_scripts/augment_by_confidence/embed_sim/analysis/plot_conf_acc.jpg'

    N = 5
    right_stat = [0] * N
    wrong_stat = [0] * N

    with open(qe_confidence_path, 'r', encoding='utf-8') as fc, \
        open(qe_gold_path, 'r', encoding='utf-8') as fg, \
        open(qe_pred_path, 'r', encoding='utf-8') as fp:
        conf_lines = fc.readlines()
        gold_lines = fg.readlines()
        pred_lines = fp.readlines()

        for conf_line, gold_line, pred_line in zip(conf_lines, gold_lines, pred_lines):
            conf_line = list(map(float, conf_line.strip('\n').split()))
            gold_line = gold_line.strip('\n').split()
            pred_line = pred_line.strip('\n').split()
            for c, g, p in zip(conf_line, gold_line, pred_line):
                if c >= 0.5 and c <= 0.6: i = 0
                elif c > 0.6 and c <= 0.7: i = 1
                elif c > 0.7 and c <= 0.8: i = 2
                elif c > 0.8 and c <= 0.9: i = 3
                elif c > 0.9 and c <= 1.0: i = 4
                if g == p: right_stat[i] += 1
                elif g != p: wrong_stat[i] += 1

    print(right_stat)
    print(wrong_stat)
    right_ratio = [right / (right + wrong) for right, wrong in zip(right_stat, wrong_stat)]

    x = np.arange(N)
    width = 0.35

    fig, ax = plt.subplots()

    p1 = ax.bar(x, right_stat, width, label='right', color='r', alpha=0.6)
    p2 = ax.bar(x, wrong_stat, width, bottom=right_stat, 
                    label='wrong', color='blue', alpha=0.3)

    ax.plot(x, right_ratio)
    # ax.axhline(0, color='grey', linewidth=0.8)
    ax.set_xlabel('QE confidence')
    ax.set_ylabel('num of right & wrong pred')
    ax.set_title('num of right & wrong pred group by QE confidence')
    ax.set_xticks(x, labels=['0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0'])
    ax.legend()

    # Label with label_type 'center' instead of the default 'edge'
    ax.bar_label(p1, label_type='center')
    ax.bar_label(p2, label_type='center')
    # ax.bar_label(bad_ratio)
    for i in range(N):
            plt.text(x[i], right_stat[i] + wrong_stat[i] + 100, str(round(right_ratio[i], 2)), ha='center')

    plt.savefig(plot_path)

plot_conf_acc()

# python3 mello_scripts/augment_by_confidence/embed_sim/analysis/plot_conf.py