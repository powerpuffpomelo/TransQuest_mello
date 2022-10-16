export CUDA_VISIBLE_DEVICES=1
lang_pair=en-zh
python3 -m mlqe_word_level.microtransquest_for_tlm -l $lang_pair
#hdfs dfs -put train_result_$lang_pair /home/byte_arnold_lq_mlnlc/user/yanyiming.mello/qe_shortcut/transquest_model

# bash mello_scripts/mitigate_memory_shortcut/tlm/train_tlm/train_tlm_en-zh.sh