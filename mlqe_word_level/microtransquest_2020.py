import os
import shutil
import argparse

from sklearn.model_selection import train_test_split

from examples.word_level.common.util import reader, prepare_testdata
from mlqe_word_level.microtransquest_config_2020 import path_prefix, TRAIN_PATH, TRAIN_SOURCE_FILE, \
    TRAIN_SOURCE_TAGS_FILE, \
    TRAIN_TARGET_FILE, \
    TRAIN_TARGET_TAGS_FLE, MODEL_TYPE, MODEL_NAME, microtransquest_config, TEMP_DIRECTORY, SEED, \
    DEV_PATH, DEV_SOURCE_FILE, DEV_TARGET_FILE, DEV_SOURCE_TAGS_FILE, DEV_TARGET_TAGS_FLE
from mlqe_word_level.run_model import MicroTransQuestModel

parser = argparse.ArgumentParser()
parser.add_argument('--lang_pair', '-l', type=str)
parser.add_argument('--model_path', type=str, default=None)
parser.add_argument('--save_name', type=str, default=None)
args = parser.parse_args()
lang_pair = args.lang_pair

TRAIN_PATH = path_prefix + lang_pair + "/train/"
DEV_PATH = path_prefix + lang_pair + "/dev/"

temp_prefix = "checkpoints/train_result_" + lang_pair
if args.save_name is not None:
    temp_prefix = "checkpoints/train_result_" + args.save_name + "_" + lang_pair
TEMP_DIRECTORY = temp_prefix + "/data"
if not os.path.exists(TEMP_DIRECTORY):
    os.makedirs(TEMP_DIRECTORY)
microtransquest_config['output_dir'] = temp_prefix + '/outputs/'
microtransquest_config['best_model_dir'] = temp_prefix + "/outputs/best_model"
microtransquest_config['cache_dir'] = temp_prefix + '/cache_dir/'

if args.model_path is not None:
    MODEL_NAME = args.model_path

raw_train_df = reader(TRAIN_PATH, TRAIN_SOURCE_FILE, TRAIN_TARGET_FILE, TRAIN_SOURCE_TAGS_FILE,
                      TRAIN_TARGET_TAGS_FLE)
raw_dev_df = reader(DEV_PATH, DEV_SOURCE_FILE, DEV_TARGET_FILE, DEV_SOURCE_TAGS_FILE,
                    DEV_TARGET_TAGS_FLE)

dev_sentences = prepare_testdata(raw_dev_df)

fold_sources_tags = []
fold_targets_tags = []

dev_fold_sources_tags = []
dev_fold_targets_tags = []

for i in range(microtransquest_config["n_fold"]):

    if os.path.exists(microtransquest_config['output_dir']) and os.path.isdir(microtransquest_config['output_dir']):
        shutil.rmtree(microtransquest_config['output_dir'])

    if microtransquest_config["evaluate_during_training"]:
        raw_train, raw_eval = train_test_split(raw_train_df, test_size=0.1, random_state=SEED * i)
        model = MicroTransQuestModel(MODEL_TYPE, MODEL_NAME, labels=["OK", "BAD"], args=microtransquest_config)
        model.train_model(raw_train, eval_data=raw_eval)
        model = MicroTransQuestModel(MODEL_TYPE, microtransquest_config["best_model_dir"], labels=["OK", "BAD"],
                                     args=microtransquest_config)

    else:
        model = MicroTransQuestModel(MODEL_TYPE, MODEL_NAME, labels=["OK", "BAD"], args=microtransquest_config)
        model.train_model(raw_train_df)
