# conda activate layoutlmv3
path=$(pwd)
echo "Current working directory : $path"
heuristic='nearest'
deg=4
hd=4
for sz in 4
do
 for sd in 0
  do
    cd $path
    echo "===== FUNSD LAGER Heuristic=$heuristic Few-Shot Size: $sz Seed: $sd"
#   --master_port 4398
   CUDA_VISIBLE_DEVICES=3 python examples/run_funsd.py  \
    --do_train --do_eval --do_predict --metric_for_best_model f1 --model_name_or_path microsoft/layoutlmv3-base  \
   --output_dir results/test-layoutlmv3-gat-$heuristic-$sz-$sd --segment_level_layout 1 --visual_embed 1 --input_size 224 \
   --max_steps 2000 --learning_rate 1e-5  \
   --load_best_model_at_end --metric_for_best_model f1 --save_strategy steps --save_steps 200 --save_strategy steps --save_total_limit 1  \
  --evaluation_strategy steps --eval_steps 200 --logging_steps 100 --logging_first_step --logging_strategy steps \
   --per_device_train_batch_size 2 --dataloader_num_workers 8  --warmup_ratio 0.1 --gradient_accumulation_steps 1  --fp16  --sz-$sz \
   --sd-$sd --hd-$hd --heuristic-$heuristic --deg-$deg $path/data/path_config.json

  done
done
