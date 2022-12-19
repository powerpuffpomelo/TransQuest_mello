export CUDA_VISIBLE_DEVICES=1
lang_pair=en-de
loss_type=logit_adjustment
save_name=logit_adjustment_loss_tau0.3_ok0.7
python3 -m mlqe_word_level.microtransquest_2021 -l $lang_pair --loss_type $loss_type --save_name $save_name
# hdfs dfs -put train_result_$lang_pair /home/byte_arnold_lq_mlnlc/user/yanyiming.mello/qe_shortcut/transquest_model

# bash mello_scripts/train/train_2021/train_en-de.sh