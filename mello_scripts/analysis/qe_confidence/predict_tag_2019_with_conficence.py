# 预测时，存储模型信心
import os
from examples.word_level.common.util import reader, prepare_testdata
from mlqe_word_level.microtransquest_config_2019 import MODEL_TYPE, microtransquest_config, TEST_SOURCE_FILE, TEST_TARGET_FILE
from mlqe_word_level.run_model import MicroTransQuestModel

lang_pair = 'en-de'
save_name = '2019'
TEST_PATH = "/opt/tiger/fake_arnold/qe_data/wmt-qe-2019-data/test_en-de"
microtransquest_config['best_model_dir'] = '/opt/tiger/fake_arnold/TransQuest_mello/checkpoints/train_result_2021_en-de/outputs/best_model'
RESULT_DIRECTORY = '/opt/tiger/fake_arnold/TransQuest_mello/checkpoints/train_result_2021_en-de/prediction_with_confidence'
if not os.path.exists(RESULT_DIRECTORY):
    os.makedirs(RESULT_DIRECTORY)

TEST_SOURCE_TAGS_FILE = "test" + save_name + ".src_tag.pred"
TEST_TARGET_TAGS_FILE = "test" + save_name + ".mtgap_tag.pred"
TEST_MT_TAGS_FILE = "test" + save_name + ".mt_tag.pred"
TEST_GAP_TAGS_FILE = "test" + save_name + ".gap_tag.pred"

test_src_conf_file = "test" + save_name + ".src_conf.pred"
test_tgt_conf_file = "test" + save_name + ".mtgap_conf.pred"
test_mt_conf_file = "test" + save_name + ".mt_conf.pred"
test_gap_conf_file = "test" + save_name + ".gap_conf.pred"

raw_test_df = reader(TEST_PATH, TEST_SOURCE_FILE, TEST_TARGET_FILE)
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


# python3 mello_scripts/analysis/qe_confidence/predict_tag_2019_with_conficence.py