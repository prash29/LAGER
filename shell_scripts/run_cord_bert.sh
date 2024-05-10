# conda activate layoutlmft
path='/home/prashant/unilm/layoutlmv3'
deg=4
for sz in 1 3 5 7 9
do
 for sd in 0 1 2 3 4 5
  do
   for hd in 4 
    do
     cd $path
     echo "==== CORD - GAT - sz-$sz-sd-$sd ======"

   CUDA_VISIBLE_DEVICES=0 python examples/run_cord.py  \
       --do_train --do_eval --do_predict --model_name_or_path microsoft/layoutlm-base-uncased \
   --output_dir results/test-lm-cord-fs-$sz-$sd \
      --save_steps 500 --save_strategy steps --save_total_limit 1  --learning_rate 1e-5\
  --evaluation_strategy steps --eval_steps 500 --logging_steps 100 --logging_first_step --logging_strategy steps --max_steps 1000  \
   --per_device_train_batch_size 2 --dataloader_num_workers 8 --warmup_ratio 0.1 --gradient_accumulation_steps 1 --fp16 --sz-$sz --sd-$sd --hd-$hd --heuristic-baseline
# --load_best_model_at_end --metric_for_best_model f1
   done
  done
done
