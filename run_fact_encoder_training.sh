

set -x -e

run_id=$(date +%s)
echo "RUN ID: $run_ts"

echo "START TIME: $(date)"


ROOT_DIR_BASE=/home/ubuntu/cloudfs/saved_models/qlora_cn
OUTPUT_PATH=$ROOT_DIR_BASE/output_$run_id

mkdir -p $OUTPUT_PATH



# based on test in ./test_cn_dataset_lenghts.py :

#source len @qt0.8: 188.0
#target len @qt0.8: 222.0
#source len @qt0.85: 228.0
#target len @qt0.85: 267.0
#source len @qt0.9: 297.0
#target len @qt0.9: 342.0
#source len @qt0.95: 396.0
#target len @qt0.95: 491.0
#source len @qt0.98: 515.0
#target len @qt0.98: 670.2800000000279




python qlora_fact_encoder.py --dataset="/home/ubuntu/qlora/metacritics_fact_encoder_training_data.pickle" \
    --dataset_format="fact_encoder" `#alpaca-clean has similar format to chinese training dataset` \
    --learning_rate 0.0001 `# QLoRA paper appendix B Table 9 `\
    --per_device_train_batch_size 1 `# fix for fitting mem `\
    --gradient_accumulation_steps 16 `# QLoRA paper appendix B Table 9  `\
    --max_steps 450 `#10 for debug, 45*5 for formal QLoRA paper appendix B Table 9, follow paper setting even though cn data is 690k much bigger than OASST1 9k, batch size considering accum 45*5`\
    --model_name_or_path "timdettmers/guanaco-33b-merged" \
    --source_max_len 42  `# default setting in code, cn model 2048 too long  `\
    --target_max_len 2829 `# follow QLoRA paper appendix B Table 9 `\
    --eval_dataset_size 1 `# mainly for testing, no need to be big` \
    --do_eval \
    --evaluation_strategy "steps" \
    --eval_steps 45 `# 10 for debug mode only, 200 for training`  \
    --output_dir $OUTPUT_PATH \
    --report_to 'wandb' \
    --sample_generate `# test sample generation every once a while`  \
    --save_steps 45 `# 20 for debug mode only, 200 for training` \
    --train_on_source true
#    --debug_mode `# only set when it's debug mode` \
