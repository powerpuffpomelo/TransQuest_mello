import os
import argparse
from mlqe_word_level.utils import reader, prepare_testdata
from mlqe_word_level.microtransquest_config import MODEL_TYPE, microtransquest_config, TEST_SOURCE_FILE, TEST_TARGET_FILE
from mlqe_word_level.run_model import MicroTQForTLMQE

parser = argparse.ArgumentParser()
parser.add_argument('--lang_pair', '-l', type=str, default='en-de')
parser.add_argument('--save_name', type=str, default='')
args = parser.parse_args()

lang_pair = args.lang_pair
TEST_PATH = "/opt/tiger/fake_arnold/TransQuest_mello/data/test/" + lang_pair + "-test20/"
temp_prefix = temp_prefix = "checkpoints/train_result_" + args.save_name + "_" + lang_pair
microtransquest_config['best_model_dir'] = temp_prefix + "/outputs/best_model"
RESULT_DIRECTORY = temp_prefix + "/prediction"
if not os.path.exists(RESULT_DIRECTORY):
    os.makedirs(RESULT_DIRECTORY)
TEST_SOURCE_TAGS_FILE = "test20.src_tag.pred"
TEST_TARGET_TAGS_FILE = "test20.mtgap_tag.pred"
TEST_MT_TAGS_FILE = "test20.mt_tag.pred"
TEST_GAP_TAGS_FILE = "test20.gap_tag.pred"

raw_test_df = reader(TEST_PATH, TEST_SOURCE_FILE, TEST_TARGET_FILE)
test_sentences = prepare_testdata(raw_test_df)

model = MicroTQForTLMQE(MODEL_TYPE, microtransquest_config["best_model_dir"], labels=["OK", "BAD"],
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


# python3 mello_scripts/mitigate_memory_shortcut/tlm/tlm_and_qe/predict_tag.py --lang_pair en-zh --save_name finetune_qe_from_tlm_lr4e-5_mask0.3