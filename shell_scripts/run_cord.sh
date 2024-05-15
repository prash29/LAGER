# conda activate layoutlmft
path='/home/prashant/unilm/layoutlmv3'
heuristic='nearest'
deg=4
for sz in 4
do
 for sd in 0
  do
   for hd in 4 
    do
     cd $path
     echo "==== CORD - GAT - sz-$sz-sd-$sd ======"

   CUDA_VISIBLE_DEVICES=0 python examples/run_cord.py  \
       --do_train --do_eval --do_predict --metric_for_best_model f1 --model_name_or_path microsoft/layoutlmv3-base  \
   --output_dir results/test-layoutlmv3-cord-gat-angles-full --segment_level_layout 1 --visual_embed 1 --input_size 224 \
     --load_best_model_at_end --metric_for_best_model f1 --save_steps 2000 --save_strategy steps --save_total_limit 1  --learning_rate 1e-5\
  --evaluation_strategy steps --eval_steps 500 --logging_steps 100 --logging_first_step --logging_strategy steps --max_steps 1000  \
   --per_device_train_batch_size 2 --dataloader_num_workers 8 --warmup_ratio 0.1 --gradient_accumulation_steps 1 --fp16 --sz-$sz --sd-$sd --hd-$hd --heuristic-$heuristic --deg-$deg /home/prashant/lager_repo/data/path_config_cord.json

   done
  done
done
