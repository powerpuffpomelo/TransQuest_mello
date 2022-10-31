# ======================= test21 ======================= #
# data_prefix=/opt/tiger/fake_arnold/mlqe-pe-21/data/test21_goldlabels

# cd $data_prefix
# for lp in en-de en-zh et-en ne-en ro-en ru-en si-en; do
#     tar zxvf ${data_prefix}/${lp}-test21.tar.gz
# done

# ======================= 20 ======================= #
# data_prefix=/opt/tiger/fake_arnold/wmt-qe-2020-data

# cd $data_prefix
# for lp in ende enzh; do
#     for split in traindev blindtest; do
#         tar zxvf ${data_prefix}/qe-${lp}-${split}.tar.gz
#     done
# done

# ======================= 19 ======================= #
# data_prefix=/opt/tiger/fake_arnold/wmt-qe-2019-data

# cd $data_prefix
# for lp in en-de; do
#     for split in test; do
#         tar zxvf ${data_prefix}/task1_${lp}_${split}.tar.gz
#     done
# done

# ======================= 17 ======================= #
data_prefix=/opt/tiger/fake_arnold/wmt-qe-2017-data

cd $data_prefix
for lp in en-de; do
    for split in test training-dev; do
        tar zxvf ${data_prefix}/task2_${lp}_${split}.tar.gz
    done
done

# ======================= 22 ======================= #
# data_prefix=/opt/tiger/fake_arnold/wmt-qe-2022-data/train-dev_data/task1_word-level/train/en-de

# cd $data_prefix
# for lp in en-de; do
#     for split in train.2021 dev.2021 test.2020; do
#         tar zxvf ${data_prefix}/${lp}-${split}.tar.gz
#     done
# done

# bash /opt/tiger/fake_arnold/TransQuest_mello/mello_scripts/data_process/data_process.sh