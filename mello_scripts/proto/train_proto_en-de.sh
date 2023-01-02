export CUDA_VISIBLE_DEVICES=1
lang_pair=en-de
model_path=checkpoints/train_result_2021_en-de/outputs/best_model
save_name=finetune
python3 -m mlqe_word_level.microtransquest_proto_2021 -l $lang_pair --model_path $model_path --save_name $save_name

# bash mello_scripts/proto/train_proto_en-de.sh
