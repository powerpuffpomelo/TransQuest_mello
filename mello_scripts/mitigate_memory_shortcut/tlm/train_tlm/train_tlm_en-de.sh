export CUDA_VISIBLE_DEVICES=5
lang_pair=en-de
save_name=checkpoints/TQ_for_TLM_dynamic_lr5e-6_mask0.15_patience30
python3 -m mlqe_word_level.microtransquest_for_tlm -l $lang_pair --save_name $save_name
#hdfs dfs -put train_result_$lang_pair /home/byte_arnold_lq_mlnlc/user/yanyiming.mello/qe_shortcut/transquest_model

# bash mello_scripts/mitigate_memory_shortcut/tlm/train_tlm/train_tlm_en-de.sh
