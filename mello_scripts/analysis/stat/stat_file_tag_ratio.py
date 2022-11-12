# 统计文件中，ok/bad比例

tag_path = '/opt/tiger/fake_arnold/wmt-qe-2019-data/train_en-de/train.mt_tag'
with open(tag_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    ok_cnt = 0
    all_cnt = 0
    for line in lines:
        line = line.strip('\n').split()
        for tag in line:
            if tag == 'OK': ok_cnt += 1
            all_cnt += 1

print('ok_ratio = %.4f' % (ok_cnt / all_cnt))

# python3 /opt/tiger/fake_arnold/TransQuest_mello/mello_scripts/analysis/stat/stat_file_tag_ratio.py