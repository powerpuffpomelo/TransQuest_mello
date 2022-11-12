# 得到训练数据词典，每个token是键，维护在训练数据gold tag中的ok次数、bad次数、ok占比、作为okbad出现的位置
import json

data_prefix = '/opt/tiger/fake_arnold/mlqe-pe_word_level_with_analysis/test21/en-de-test21/'
mt_path = data_prefix + 'test2021.tok.mt'
mt_tag_path = data_prefix + 'test2021.mt_tag'
save_path = data_prefix + 'test2021.token_mt_tag_stat.json'

def stat_token_train_tag(mt_path, tag_path, save_path):
    with open(mt_path, 'r', encoding='utf-8') as f_mt, open(tag_path, 'r', encoding='utf-8') as f_tag, \
        open(save_path, 'w', encoding='utf-8') as f_save:
        stat_dict = dict()
        id = 0
        for line_mt, line_tag in zip(f_mt.readlines(), f_tag.readlines()):
            line_mt = line_mt.strip('\n').split()
            line_tag = line_tag.strip('\n').split()
            word_id = 0
            for token, tag in zip(line_mt, line_tag):
                if token not in stat_dict:
                    stat_dict[token] = {'train_freq':0, 'ok_freq':0, 'bad_freq':0, 'ok_ratio':0}
                stat_dict[token]['train_freq'] += 1
                if tag == 'OK': 
                    stat_dict[token]['ok_freq'] += 1
                else:
                    stat_dict[token]['bad_freq'] += 1
                word_id += 1
            id += 1
        for word, stat in stat_dict.items():
            stat['ok_ratio'] = stat['ok_freq'] / stat['train_freq']
        json.dump(stat_dict, f_save, indent=1, ensure_ascii=False)

stat_token_train_tag(mt_path, mt_tag_path, save_path)

# python3 mello_scripts/analysis/stat/stat_token_train_tag.py