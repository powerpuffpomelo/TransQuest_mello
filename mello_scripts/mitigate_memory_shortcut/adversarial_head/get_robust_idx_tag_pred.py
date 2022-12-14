# 只得到idx处的预测tag存起来，其余tag不care

# lang_pairs = ['en-de', 'en-zh', 'et-en', 'ne-en', 'ro-en', 'ru-en', 'si-en']
lang_pairs = ['en-de']
adv_lambdas = [-0.01, -0.05, -0.1, -0.3, -0.5, -0.6, -0.7, -1.0]
parts = ['popular', 'niche', 'same', 'unseen', 'same_and_unseen']

for lang_pair in lang_pairs:
    for adv_lambda in adv_lambdas:
        # test_prefix = "transquest_model/train_result_" + lang_pair + "/prediction/"
        test_prefix = "train_result_memory_shortcut_adv_" + lang_pair + "_adv_limit_5_lambda_" + str(adv_lambda) + "/prediction/"
        all_tag_path = test_prefix + "test20.mt_tag.pred"
        for part in parts:
            robust_idx_path = "data/test/" + lang_pair + "-test20/test20.mt_" + part + "_idx"
            robust_tag_path = test_prefix + "test20.mt_tag_" + part + "_part.pred"

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

# python3 mello_scripts/train_memory_shortcut/get_robust_idx_tag_pred.py