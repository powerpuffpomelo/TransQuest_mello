import os
from examples.word_level.common.util import reader, prepare_testdata
from mlqe_word_level.microtransquest_config.microtransquest_config_2021 import MODEL_TYPE, microtransquest_config, TEST_SOURCE_FILE, TEST_TARGET_FILE
from transquest.algo.word_level.microtransquest.run_model import MicroTransQuestModel

lang_pair = 'en-de'
save_name = '21'
TEST_PATH = "/opt/tiger/fake_arnold/qe_data/qe_data_mello/test21/" + lang_pair + "-test21"
pred_model_name = '21_focal_loss'
microtransquest_config['best_model_dir'] = "checkpoints/train_result_" + pred_model_name + "_" + lang_pair + "/outputs/best_model"
RESULT_DIRECTORY = "checkpoints/train_result_" + pred_model_name + "_" + lang_pair + "/prediction"
if not os.path.exists(RESULT_DIRECTORY):
    os.makedirs(RESULT_DIRECTORY)
TEST_SOURCE_TAGS_FILE = "test" + save_name + ".src_tag.pred"
TEST_TARGET_TAGS_FILE = "test" + save_name + ".mtgap_tag.pred"
TEST_MT_TAGS_FILE = "test" + save_name + ".mt_tag.pred"
TEST_GAP_TAGS_FILE = "test" + save_name + ".gap_tag.pred"

raw_test_df = reader(TEST_PATH, TEST_SOURCE_FILE, TEST_TARGET_FILE)
test_sentences = prepare_testdata(raw_test_df)

model = MicroTransQuestModel(MODEL_TYPE, microtransquest_config["best_model_dir"], labels=["OK", "BAD"],
                            args=microtransquest_config)

sources_tags, targets_tags = model.predict(test_sentences, split_on_space=True)

with open(os.path.join(RESULT_DIRECTORY, TEST_SOURCE_TAGS_FILE), 'w', encoding='utf-8') as f:
    for src_tag_line in sources_tags:
        f.write(' '.join(src_tag_line) + '\n')

with open(os.path.join(RESULT_DIRECTORY, TEST_TARGET_TAGS_FILE), 'w', encoding='utf-8') as f, \
    open(os.path.join(RESULT_DIRECTORY, TEST_MT_TAGS_FILE), 'w', encoding='utf-8') as fm, \
    open(os.path.join(RESULT_DIRECTORY, TEST_GAP_TAGS_FILE), 'w', encoding='utf-8') as fg:
    for tgt_tag_line in targets_tags:
        f.write(' '.join(tgt_tag_line) + '\n')
        strm = ""
        strg = ""
        for i, tag in enumerate(tgt_tag_line):
            if i % 2 == 1:
                strm = strm + tag + ' '
            else:
                strg = strg + tag + ' '
        fm.write(strm.strip() + '\n')
        fg.write(strg.strip() + '\n')


# python3 /opt/tiger/fake_arnold/TransQuest_mello/mello_scripts/train/train_2021/predict_tag_2021.py