from multiprocessing import cpu_count

lang_pair = 'en-zh'

TRAIN_PATH = "data/train/" + lang_pair + "-train/"
TRAIN_SOURCE_FILE = "train.src"
TRAIN_SOURCE_TAGS_FILE = "train.source_tags"
TRAIN_TARGET_FILE = "train.mt"
TRAIN_TARGET_TAGS_FLE = "train.tags"
TRAIN_PE_FILE = "train.pe"
TRAIN_SOURCE_ADV_TAGS_FILE = "train.src_ok_ratio_tag"
TRAIN_TARGET_ADV_TAGS_FILE = "train.mtgap_ok_ratio_tag"

DEV_PATH = "data/dev/" + lang_pair + "-dev/"
DEV_SOURCE_FILE = "dev.src"
DEV_SOURCE_TAGS_FILE = "dev.source_tags"
DEV_TARGET_FILE = "dev.mt"
DEV_TARGET_TAGS_FLE = "dev.tags"
DEV_PE_FILE = "dev.pe"
DEV_SOURCE_ADV_TAGS_FILE = "dev.src_ok_ratio_tag"
DEV_TARGET_ADV_TAGS_FILE = "dev.mtgap_ok_ratio_tag"

TEST_PATH = "data/test/" + lang_pair + "-test20/"
TEST_SOURCE_FILE = "test20.src"
TEST_TARGET_FILE = "test20.mt"

TEST_SOURCE_TAGS_FILE = "test_predictions_src.txt"
TEST_TARGET_TAGS_FLE = "test_predictions_mt.txt"

DEV_SOURCE_TAGS_FILE_SUB = "dev_predictions_src.txt"
DEV_TARGET_TAGS_FILE_SUB = "dev_predictions_mt.txt"

SEED = 777
TEMP_DIRECTORY = "train_result_" + lang_pair + "/data"
GOOGLE_DRIVE = False
DRIVE_FILE_ID = None
MODEL_TYPE = "xlmroberta"
MODEL_NAME = "transformers/xlm-roberta-large-for-tc"

microtransquest_config = {
    'output_dir': 'train_result_' + lang_pair + '/outputs/',
    "best_model_dir": "train_result_" + lang_pair + "/outputs/best_model",
    'cache_dir': 'train_result_' + lang_pair + '/cache_dir/',

    'fp16': False,
    'fp16_opt_level': 'O1',
    'max_seq_length': 200,
    'train_batch_size': 8,
    'gradient_accumulation_steps': 1,
    'eval_batch_size': 8,
    'num_train_epochs': 3,
    'weight_decay': 0,
    'learning_rate': 2e-5,
    'adam_epsilon': 1e-8,
    'warmup_ratio': 0.1,
    'warmup_steps': 0,
    'max_grad_norm': 1.0,
    'do_lower_case': False,

    'logging_steps': 300,
    'save_steps': 300,
    "no_cache": False,
    "no_save": False,
    "save_recent_only": False,
    'save_model_every_epoch': False,
    'n_fold': 3,
    'evaluate_during_training': True,
    "evaluate_during_training_silent": True,
    'evaluate_during_training_steps': 300,
    "evaluate_during_training_verbose": True,
    'use_cached_eval_features': False,
    "save_best_model": True,
    'save_eval_checkpoints': True,
    'tensorboard_dir': None,
    "save_optimizer_and_scheduler": True,

    'regression': True,

    'overwrite_output_dir': True,
    'reprocess_input_data': True,

    'process_count': cpu_count() - 2 if cpu_count() > 2 else 1,
    'n_gpu': 1,
    'use_multiprocessing': True,
    "multiprocessing_chunksize": 500,
    'silent': False,

    'wandb_project': None,
    'wandb_kwargs': {},

    "use_early_stopping": True,
    "early_stopping_patience": 10,
    "early_stopping_delta": 0,
    "early_stopping_metric": "eval_loss",
    "early_stopping_metric_minimize": True,
    "early_stopping_consider_epochs": False,

    "manual_seed": SEED,

    "add_tag": False,
    "tag": "_",

    "default_quality": "OK",

    "config": {},
    "local_rank": -1,
    "encoding": None,

    "source_column": "source",
    "target_column": "target",
    "source_tags_column": "source_tags",
    "target_tags_column": "target_tags",

    "qe_lambda": 0.8,
}
