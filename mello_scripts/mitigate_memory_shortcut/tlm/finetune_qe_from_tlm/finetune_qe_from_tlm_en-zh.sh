export CUDA_VISIBLE_DEVICES=1
lang_pair=en-zh
model_path=/opt/tiger/fake_arnold/TransQuest_mello/train_result_TQ_for_TLM_dynamic_lr4e-5_en-zh/outputs/checkpoint_4800
save_name=finetune_qe_from_tlm_lr4e-5_mask0.3
python3 -m mlqe_word_level.microtransquest -l $lang_pair --model_path $model_path --save_name $save_name
#hdfs dfs -put train_result_$lang_pair /home/byte_arnold_lq_mlnlc/user/yanyiming.mello/qe_shortcut/transquest_model

# bash mello_scripts/mitigate_memory_shortcut/tlm/finetune_qe_from_tlm/finetune_qe_from_tlm_en-zh.sh