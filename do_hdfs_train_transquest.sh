# 读取命令
echo "#=========================== print cmd ===========================#"
lang_pair=`echo "$@" | awk -F'+' '{print $1}'`  # en-zh
echo ${lang_pair}

# 复制数据
echo "#=========================== get data ===========================#"
hdfs dfs -get /home/byte_arnold_lq_mlnlc/user/yanyiming.mello/data/mlqe-pe_word_level_with_analysis ./
mv mlqe-pe_word_level_with_analysis data

# 复制xlm roberta预训练模型
echo "#=========================== get xlmr ===========================#"
mkdir -p transformers
hdfs dfs -get /home/byte_arnold_lq_mlnlc/user/yanyiming.mello/model/transformers/xlm-roberta ./transformers

# 训练啦
echo "#=========================== begin training ===========================#"
python3 -m mlqe_word_level.microtransquest -l $lang_pair # en-zh

# 训完上传输出到hdfs啦
echo "Finish training, uploading outputs to hdfs"
mv train_result train_result_$lang_pair
hdfs dfs -put train_result_$lang_pair /home/byte_arnold_lq_mlnlc/user/yanyiming.mello/qe_shortcut/transquest_model
echo "Finish uploading outputs to hdfs"
