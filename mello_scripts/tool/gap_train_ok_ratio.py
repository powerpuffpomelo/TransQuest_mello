# 统计gap在训练集上的ok ratio，并加到ok_ratio_tag标注文件中

lang_pairs = ['en-de', 'en-zh', 'et-en', 'ne-en', 'ro-en', 'ru-en', 'si-en']

for lang_pair in lang_pairs:
    train_gap_tag_path = 'data/train/' + lang_pair + '-train/train.gap_tag'
    test_prefix = "data/train/" + lang_pair + "-train/"
    qe_ok_ratio_tag_mt = test_prefix + "train.mt_ok_ratio_tag"
    qe_ok_ratio_tag_mtgap = test_prefix + "train.mtgap_ok_ratio_tag"
    with open(train_gap_tag_path, 'r', encoding='utf-8') as f:
        ok_cnt = 0
        all_cnt = 0
        for line in f.readlines():
            line = line.strip('\n').split()
            for tag in line:
                if tag == "OK": ok_cnt += 1
                all_cnt += 1
        gap_ok_ratio = ok_cnt / all_cnt
        print('%s ok_ratio: %f' % (lang_pair, gap_ok_ratio))
    with open(qe_ok_ratio_tag_mt, 'r', encoding='utf-8') as f_mt_ok, \
        open(qe_ok_ratio_tag_mtgap, 'w', encoding='utf-8') as f_mtgap_ok:
        for line_mt_ok in f_mt_ok.readlines():
            line_mt_ok = line_mt_ok.strip('\n').split()
            line_mtgap_ok = [gap_ok_ratio]
            for token_mt_ok in line_mt_ok:
                line_mtgap_ok.append(float(token_mt_ok))
                line_mtgap_ok.append(gap_ok_ratio)
            assert len(line_mtgap_ok) == len(line_mt_ok) * 2 + 1
            f_mtgap_ok.write(' '.join(map(str, line_mtgap_ok)) + '\n')

# python3 mello_scripts/tool/gap_train_ok_ratio.py