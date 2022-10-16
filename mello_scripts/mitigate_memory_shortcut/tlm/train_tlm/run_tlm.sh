data_prefix=/opt/tiger/fake_arnold/TransQuest_mello/data/test/en-zh-toy-test20
python3 mello_scripts/mitigate_memory_shortcut/tlm/train_tlm/run_tlm.py \
    --model_name_or_path xlm-roberta-large \
    --train_src $data_prefix/test20.src \
    --train_tgt $data_prefix/test20.mt \
    --val_src $data_prefix/test20.src \
    --val_tgt $data_prefix/test20.mt \
    --batch_size 1 \
    --optimizer AdamW \
    --learning_rate 5e-5 \
    --device_id 0 \
    --max_epochs 2 \
    --eval_steps 10 \
    --output_dir tmp/test-tlm \
    --save_name tlm 

# bash mello_scripts/mitigate_memory_shortcut/tlm/train_tlm/run_tlm.sh