# tag2idx
# 把tag转换成idx，比如挑出部分tag记录idx
#"""
lang_pair = 'si-en'
test_prefix = "data/test/" + lang_pair + "-test20/"
qe_test_niche_tag_path = test_prefix + "test20.mt_niche_tag"
qe_test_niche_idx_path = test_prefix + "test20.mt_niche_idx"         # 小众词汇的idx 0
qe_test_popular_idx_path = test_prefix + "test20.mt_popular_idx"     # 大众词汇的idx 1
qe_test_same_idx_path = test_prefix + "test20.mt_same_idx"           # 训练集中标签比例okbad各半的词汇的idx 0.5
qe_test_unseen_idx_path = test_prefix + "test20.mt_unseen_idx"       # 没在训练集中出现过的词汇的idx -1
qe_test_same_and_unseen_idx_path = test_prefix + "test20.mt_same_and_unseen_idx"   # 前两个集合加起来

with open(qe_test_niche_tag_path, 'r', encoding='utf-8') as ftag, open(qe_test_same_and_unseen_idx_path, 'w', encoding='utf-8') as fidx:
    for tag_line in ftag.readlines():
        idx_list = []
        tag_line = tag_line.strip('\n').split()
        for i, tag in enumerate(tag_line):
            if tag == '-1' or tag == '0.5':
                idx_list.append(i)
        fidx.write(' '.join(map(str, idx_list)) + '\n')
#"""
# idx2tag
"""
mt_path = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/train.mt.BPE"
idx_path = "/data1/yanym/data/robust_train_memory/train.niche_idx.BPE"
save_path = "/data1/yanym/data/robust_boosting/train.niche_tag.BPE"

with open(mt_path, 'r', encoding='utf-8') as fmt, open(idx_path, 'r', encoding = 'utf-8') as fidx, \
    open(save_path, 'w', encoding='utf-8') as fs:
    mt_lines = fmt.readlines()
    idx_lines = fidx.readlines()
    for mt_line, idx_line in zip(mt_lines, idx_lines):
        mt_line = mt_line.strip('\n').split()
        idx_line = idx_line.strip('\n').split()
        idx_line = [int(x) for x in idx_line]
        tag_list = []
        for i in range(len(mt_line)):
            if i in idx_line:
                tag_list.append(1)
            else:
                tag_list.append(0)
        fs.write(' '.join(map(str, tag_list)) + '\n')
"""


# python3 mello_scripts/train_memory_shortcut/tag2idx.py