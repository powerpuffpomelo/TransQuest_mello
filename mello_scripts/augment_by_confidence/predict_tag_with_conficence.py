# 预测时，存储模型信心
import os
from examples.word_level.common.util import reader, prepare_testdata
from mlqe_word_level.microtransquest_config.microtransquest_config import MODEL_TYPE, microtransquest_config
from mlqe_word_level.run_model import MicroTransQuestModel

split = 'test'
pred_year = '21'

TEST_PATH = '/opt/tiger/fake_arnold/qe_data/qe_data_mello/'+ split + pred_year + '/en-de-' + split + pred_year + '/'

test_src_file = split + pred_year + '.src'
test_mt_file = split + pred_year + '.mt'


microtransquest_config['best_model_dir'] = '/opt/tiger/fake_arnold/TransQuest_mello/checkpoints/train_result_2021_en-de/outputs/best_model'
RESULT_DIRECTORY = '/opt/tiger/fake_arnold/TransQuest_mello/checkpoints/train_result_2021_en-de/prediction_with_confidence'
if not os.path.exists(RESULT_DIRECTORY):
    os.makedirs(RESULT_DIRECTORY)
# 预测结果保存文件
TEST_SOURCE_TAGS_FILE = split + pred_year + ".src_tag.pred"
TEST_TARGET_TAGS_FILE = split + pred_year + ".mtgap_tag.pred"
TEST_MT_TAGS_FILE = split + pred_year + ".mt_tag.pred"
TEST_GAP_TAGS_FILE = split + pred_year + ".gap_tag.pred"

test_src_conf_file = split + pred_year + ".src_conf.pred"
test_tgt_conf_file = split + pred_year + ".mtgap_conf.pred"
test_mt_conf_file = split + pred_year + ".mt_conf.pred"
test_gap_conf_file = split + pred_year + ".gap_conf.pred"

raw_test_df = reader(TEST_PATH, test_src_file, test_mt_file)
test_sentences = prepare_testdata(raw_test_df)

model = MicroTransQuestModel(MODEL_TYPE, microtransquest_config["best_model_dir"], labels=["OK", "BAD"],
                            args=microtransquest_config)

sources_tags, targets_tags, sources_confidence, targets_confidence = model.predict_with_confidence(test_sentences, split_on_space=True)

with open(os.path.join(RESULT_DIRECTORY, TEST_SOURCE_TAGS_FILE), 'w', encoding='utf-8') as f1, \
    open(os.path.join(RESULT_DIRECTORY, test_src_conf_file), 'w', encoding='utf-8') as f2:
    for src_tag_line, src_conf_line in zip(sources_tags, sources_confidence):
        src_conf_line = list(map(str, src_conf_line))
        f1.write(' '.join(src_tag_line) + '\n')
        f2.write(' '.join(src_conf_line) + '\n')

with open(os.path.join(RESULT_DIRECTORY, TEST_TARGET_TAGS_FILE), 'w', encoding='utf-8') as f, \
    open(os.path.join(RESULT_DIRECTORY, TEST_MT_TAGS_FILE), 'w', encoding='utf-8') as fm, \
    open(os.path.join(RESULT_DIRECTORY, TEST_GAP_TAGS_FILE), 'w', encoding='utf-8') as fg, \
    open(os.path.join(RESULT_DIRECTORY, test_tgt_conf_file), 'w', encoding='utf-8') as fc, \
    open(os.path.join(RESULT_DIRECTORY, test_mt_conf_file), 'w', encoding='utf-8') as fcm, \
    open(os.path.join(RESULT_DIRECTORY, test_gap_conf_file), 'w', encoding='utf-8') as fcg:
    for tgt_tag_line, tgt_conf_line in zip(targets_tags, targets_confidence):
        assert len(tgt_tag_line) == len(tgt_conf_line)
        
        tgt_conf_line = list(map(str, tgt_conf_line))

        f.write(' '.join(tgt_tag_line) + '\n')
        fc.write(' '.join(tgt_conf_line) + '\n')

        strm = ""
        strg = ""
        for i, tag in enumerate(tgt_tag_line):
            if i % 2 == 1:
                strm = strm + tag + ' '
            else:
                strg = strg + tag + ' '
        fm.write(strm.strip() + '\n')
        fg.write(strg.strip() + '\n')

        strm = ""
        strg = ""
        for i, conf in enumerate(tgt_conf_line):
            if i % 2 == 1:
                strm = strm + conf + ' '
            else:
                strg = strg + conf + ' '
        fcm.write(strm.strip() + '\n')
        fcg.write(strg.strip() + '\n')


# python3 mello_scripts/augment_by_confidence/predict_tag_with_conficence.py