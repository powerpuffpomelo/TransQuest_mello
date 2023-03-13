export CUDA_VISIBLE_DEVICES=1
lang_pair=en-de
save_name=qe_from_fix2100_lr1e-5_continue
python3 -m mlqe_word_level.microtransquest_2021 -l $lang_pair --save_name $save_name
# hdfs dfs -put train_result_$lang_pair /home/byte_arnold_lq_mlnlc/user/yanyiming.mello/qe_shortcut/transquest_model

# bash mello_scripts/train/train_2021/train_en-de.sh
