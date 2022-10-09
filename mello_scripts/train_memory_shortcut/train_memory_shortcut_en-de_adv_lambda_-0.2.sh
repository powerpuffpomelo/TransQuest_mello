export CUDA_VISIBLE_DEVICES=0
lang_pair=en-de
adv_lambda=-0.2
python3 -m mlqe_word_level.microtransquest_with_adv -l $lang_pair --adv_lambda $adv_lambda
hdfs dfs -put train_result_memory_shortcut_adv_${lang_pair}_adv_lambda_${adv_lambda} /home/byte_arnold_lq_mlnlc/user/yanyiming.mello/qe_shortcut/transquest_model_with_train_memory_shortcut_adv_training

# bash mello_scripts/train_memory_shortcut/train_memory_shortcut_en-de_adv_lambda_-0.2.sh