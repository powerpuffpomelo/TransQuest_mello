export CUDA_VISIBLE_DEVICES=2
lang_pair=en-de
save_name=direct
python3 -m mlqe_word_level.microtransquest_proto_2021 -l $lang_pair --save_name $save_name

# bash mello_scripts/proto/train_proto_en-de_2.sh
