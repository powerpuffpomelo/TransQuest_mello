# 给测试集每个token打上 训练时ok占比的标签，和mt每个token位置对应
# 如果没在训练集出现过，就标0.5（暂定）
import json

lang_pairs = ['en-de', 'en-zh', 'et-en', 'ne-en', 'ro-en', 'ru-en', 'si-en']

for lang_pair in lang_pairs:
    train_tag_stat_path = "data/train/" + lang_pair + "-train/train_token_src_tag_stat.json"
    test_prefix = "data/train/" + lang_pair + "-train/"
    qe_test_mt_path = test_prefix + "train.src"
    qe_test_ok_ratio_tag_path = test_prefix + "train.src_ok_ratio_tag"

    with open(train_tag_stat_path, 'r', encoding='utf-8') as f:
        train_tag_stat = json.load(f)

    with open(qe_test_mt_path, 'r', encoding='utf-8') as f_mt, open(qe_test_ok_ratio_tag_path, 'w', encoding='utf-8') as f_save:
        for line_mt in f_mt.readlines():
            line_mt = line_mt.strip('\n').split()
            line_ok_ratio = []
            for token in line_mt:
                if token in train_tag_stat:
                    line_ok_ratio.append(train_tag_stat[token]['ok_ratio'])
                else:
                    line_ok_ratio.append(-0.5)
            f_save.write(' '.join(map(str, line_ok_ratio)) + '\n')

# python3 mello_scripts/train_memory_shortcut/annotate_token_train_ok_ratio_tag.py