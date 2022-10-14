export CUDA_VISIBLE_DEVICES=1
lang_pair=en-de
adv_lambda=-0.1
adv_limit=5
python3 -m mlqe_word_level.microtransquest_with_adv -l $lang_pair --adv_limit $adv_limit --adv_lambda $adv_lambda
#hdfs dfs -put train_result_memory_shortcut_adv_${lang_pair}_adv_limit_${limit}_lambda_${adv_lambda} /home/byte_arnold_lq_mlnlc/user/yanyiming.mello/qe_shortcut/transquest_model_with_train_memory_shortcut_adv_training

# bash mello_scripts/train_memory_shortcut/train_memory_shortcut_en-de_adv_lambda_-0.1.sh