# 得到训练数据词典，每个token是键，维护在训练数据gold tag中的ok次数、bad次数、ok占比、作为okbad出现的位置
import json

lang_pairs = ['en-de', 'en-zh', 'et-en', 'ne-en', 'ro-en', 'ru-en', 'si-en']

for lang_pair in lang_pairs:
    prefix = "data/train/" + lang_pair + "-train/"
    qe_train_mt_path = prefix + "train.src"
    qe_train_gold_tag_path = prefix + "train.source_tags"

    save_path = prefix + "train_token_src_tag_stat.json"

    with open(qe_train_mt_path, 'r', encoding='utf-8') as f_mt, open(qe_train_gold_tag_path, 'r', encoding='utf-8') as f_tag, \
        open(save_path, 'w', encoding='utf-8') as f_save:
        stat_dict = dict()
        id = 0
        for line_mt, line_tag in zip(f_mt.readlines(), f_tag.readlines()):
            line_mt = line_mt.strip('\n').split()
            line_tag = line_tag.strip('\n').split()
            word_id = 0
            for token, tag in zip(line_mt, line_tag):
                if token not in stat_dict:
                    stat_dict[token] = {'train_freq':0, 'ok_freq':0, 'bad_freq':0, 'ok_ratio':0, 'ok_pos':[], 'bad_pos':[]}
                stat_dict[token]['train_freq'] += 1
                if tag == 'OK': 
                    stat_dict[token]['ok_freq'] += 1
                    stat_dict[token]['ok_pos'].append([id, word_id])
                else: 
                    stat_dict[token]['bad_freq'] += 1
                    stat_dict[token]['bad_pos'].append([id, word_id])
                word_id += 1
            id += 1
        for word, stat in stat_dict.items():
            stat['ok_ratio'] = stat['ok_freq'] / stat['train_freq']
        json.dump(stat_dict, f_save, indent=1, ensure_ascii=False)

# python3 mello_scripts/train_memory_shortcut/get_token_train_tag_stat.py