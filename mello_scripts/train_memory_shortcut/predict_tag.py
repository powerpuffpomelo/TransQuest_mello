import os
from examples.word_level.common.util import reader, prepare_testdata
from mlqe_word_level.microtransquest_config import MODEL_TYPE, microtransquest_config, TEST_SOURCE_FILE, TEST_TARGET_FILE
from transquest.algo.word_level.microtransquest.run_model import MicroTransQuestModel

lang_pair = 'en-de'
TEST_PATH = "/opt/tiger/fake_arnold/TransQuest_mello/data/test/" + lang_pair + "-test20/"
# microtransquest_config['best_model_dir'] = "train_result_" + lang_pair + "/outputs/best_model"
model_path_prefix = '/opt/tiger/fake_arnold/TransQuest_mello/train_result_memory_shortcut_adv_en-de_adv_lambda_-0.01/'
model_path = model_path_prefix + 'outputs/best_model'
RESULT_DIRECTORY = model_path_prefix + "prediction"
if not os.path.exists(RESULT_DIRECTORY):
    os.makedirs(RESULT_DIRECTORY)
TEST_SOURCE_TAGS_FILE = "test20.src_tag.pred"
TEST_TARGET_TAGS_FILE = "test20.mtgap_tag.pred"
TEST_MT_TAGS_FILE = "test20.mt_tag.pred"
TEST_GAP_TAGS_FILE = "test20.gap_tag.pred"

raw_test_df = reader(TEST_PATH, TEST_SOURCE_FILE, TEST_TARGET_FILE)
test_sentences = prepare_testdata(raw_test_df)

model = MicroTransQuestModel(MODEL_TYPE, model_path, labels=["OK", "BAD"],
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


"""
export CUDA_VISIBLE_DEVICES=7
python -m mello_scripts.predict_tag
"""
# python3 mello_scripts/train_memory_shortcut/predict_tag.py