export CUDA_VISIBLE_DEVICES=2
lang_pair=en-de
save_name=2019
python3 -m mlqe_word_level.microtransquest_2019 -l $lang_pair --save_name $save_name
# hdfs dfs -put train_result_$lang_pair /home/byte_arnold_lq_mlnlc/user/yanyiming.mello/qe_shortcut/transquest_model

# bash /opt/tiger/fake_arnold/TransQuest_mello/mello_scripts/train/train_2019/train_en-de_2019.sh