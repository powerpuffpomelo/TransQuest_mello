export CUDA_VISIBLE_DEVICES=0
lang_pair=en-zh
python3 -m mlqe_word_level.microtransquest -l $lang_pair
#hdfs dfs -put train_result_$lang_pair /home/byte_arnold_lq_mlnlc/user/yanyiming.mello/qe_shortcut/transquest_model

# bash mello_scripts/train/train_en-zh.sh