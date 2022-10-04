# 只得到idx处的预测tag存起来，其余tag不care

lang_pair = 'si-en'
test_prefix = "train_result_" + lang_pair + "/prediction/"
all_tag_path = test_prefix + "test20.mt_tag.pred"
robust_idx_path = "data/test/" + lang_pair + "-test20/test20.mt_popular_idx"
robust_tag_path = test_prefix + "test20.mt_tag_popular_part.pred"

with open(all_tag_path, 'r', encoding='utf-8') as ft, open(robust_idx_path, 'r', encoding='utf-8') as fid, \
    open(robust_tag_path, 'w', encoding='utf-8') as fsave:
    all_tag_lines = ft.readlines()
    idx_lines = fid.readlines()
    for all_tag_line, idx_line in zip(all_tag_lines, idx_lines):
        all_tag_line = all_tag_line.strip('\n').split()
        idx_line = list(map(int, idx_line.strip('\n').split()))
        robust_tag_line = []
        for i, tag in enumerate(all_tag_line):
            if i in idx_line:
                robust_tag_line.append(tag)
        fsave.write(' '.join(robust_tag_line) + '\n')

# python3 scripts/get_robust_idx_tag_pred.py