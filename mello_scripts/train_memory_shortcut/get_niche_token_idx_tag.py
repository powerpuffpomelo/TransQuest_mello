# 根据训练时大部分被打什么标签，得到测试集or训练集的小众tag文件，和mt对应，得到01文件，0代表小众，1代表大众, 0.5代表训练集中标签okbad各半，-1代表没在训练集出现过
import json

lang_pair = 'si-en'
train_tag_stat_path = "data/train/" + lang_pair + "-train/train_token_tag_stat.json"
test_prefix = "data/test/" + lang_pair + "-test20/"
qe_test_mt_path = test_prefix + "test20.mt"
qe_test_gold_tag_path = test_prefix + "test20.mt_tag"
qe_test_niche_tag_path = test_prefix + "test20.mt_niche_tag"

with open(train_tag_stat_path, 'r', encoding='utf-8') as f:
    train_tag_stat = json.load(f)

with open(qe_test_mt_path, 'r', encoding='utf-8') as f_mt, open(qe_test_gold_tag_path, 'r', encoding='utf-8') as f_tag, \
    open(qe_test_niche_tag_path, 'w', encoding='utf-8') as f_save:
    for line_mt, line_tag in zip(f_mt.readlines(), f_tag.readlines()):
        line_mt = line_mt.strip('\n').split()
        line_tag = line_tag.strip('\n').split()
        line_niche = []
        for token, tag in zip(line_mt, line_tag):
            if token in train_tag_stat:
                if (tag == 'OK' and train_tag_stat[token]['ok_ratio'] < 0.5) or (tag == 'BAD' and train_tag_stat[token]['ok_ratio'] > 0.5): line_niche.append(0)
                elif (tag == 'OK' and train_tag_stat[token]['ok_ratio'] > 0.5) or (tag == 'BAD' and train_tag_stat[token]['ok_ratio'] < 0.5): line_niche.append(1)
                else:
                    assert train_tag_stat[token]['ok_ratio'] == 0.5
                    line_niche.append(0.5)
            else:
                line_niche.append(-1)
        f_save.write(' '.join(map(str, line_niche)) + '\n')

# python3 mello_scripts/train_memory_shortcut/get_niche_token_idx_tag.py