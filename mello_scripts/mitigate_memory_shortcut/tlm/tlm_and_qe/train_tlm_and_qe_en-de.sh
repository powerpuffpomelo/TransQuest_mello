export CUDA_VISIBLE_DEVICES=6
lang_pair=en-de
save_name=tlm_and_qe_qelambda0.5
python3 -m mlqe_word_level.microtransquest_for_tlm_qe -l $lang_pair --save_name $save_name
#hdfs dfs -put train_result_$lang_pair /home/byte_arnold_lq_mlnlc/user/yanyiming.mello/qe_shortcut/transquest_model

# bash mello_scripts/mitigate_memory_shortcut/tlm/tlm_and_qe/train_tlm_and_qe_en-de.sh