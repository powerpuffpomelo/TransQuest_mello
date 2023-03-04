import os
import shutil
import argparse

from sklearn.model_selection import train_test_split

from mlqe_word_level.utils import reader, prepare_testdata
from mlqe_word_level.microtransquest_config.microtransquest_config_for_tlm import MODEL_TYPE, MODEL_NAME, microtransquest_config, \
    path_prefix, TRAIN_PATH, TRAIN_SOURCE_FILE, TRAIN_PE_FILE, \
    DEV_PATH, DEV_SOURCE_FILE, DEV_PE_FILE, \
    TEMP_DIRECTORY
from mlqe_word_level.run_model import MicroTQForTLM

parser = argparse.ArgumentParser()
parser.add_argument('--lang_pair', '-l', type=str)
parser.add_argument('--save_name', type=str, default='TQ_for_TLM')
args = parser.parse_args()
lang_pair = args.lang_pair

TRAIN_PATH = path_prefix + "train21/" + lang_pair + "-train/"
DEV_PATH = path_prefix + "dev21/" + lang_pair + "-dev/"
TEST_PATH = path_prefix + "test21/" + lang_pair + "-test21/"
temp_prefix = "train_result_" + args.save_name + "_" + lang_pair
if not os.path.exists(temp_prefix):
    os.makedirs(temp_prefix)
TEMP_DIRECTORY = temp_prefix + "/data"
microtransquest_config['output_dir'] = temp_prefix + '/outputs/'
microtransquest_config['best_model_dir'] = temp_prefix + "/outputs/best_model"
microtransquest_config['cache_dir'] = temp_prefix + '/cache_dir/'

raw_train_df = reader(path=TRAIN_PATH, source_file=TRAIN_SOURCE_FILE, target_file=TRAIN_PE_FILE)
raw_dev_df = reader(path=DEV_PATH, source_file=DEV_SOURCE_FILE, target_file=DEV_PE_FILE)
# raw_test_df = reader(path=TEST_PATH, source_file=TEST_SOURCE_FILE, target_file=TEST_PE_FILE)

if os.path.exists(microtransquest_config['output_dir']) and os.path.isdir(microtransquest_config['output_dir']):
    shutil.rmtree(microtransquest_config['output_dir'])

if microtransquest_config["evaluate_during_training"]:
    model = MicroTQForTLM(MODEL_TYPE, MODEL_NAME, args=microtransquest_config)
    model.train_model(raw_train_df, eval_data=raw_dev_df)

else:
    model = MicroTQForTLM(MODEL_TYPE, MODEL_NAME, args=microtransquest_config)
    model.train_model(raw_train_df)
