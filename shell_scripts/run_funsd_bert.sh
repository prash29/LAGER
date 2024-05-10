# conda activate layoutlmft
path='/home/prashant/unilm/layoutlmv3'
heuristic='baseline'
# echo $heuristic > 
deg=4
theta=60
for sz in 2 4 6 8 10
do
 for sd in 0 1 2 3 4 5
  do
  for hd in 4
   do
     cd $path
     echo "===== FUNSD gat angles-sz-$sz-sd-$sd ===="
#   --master_port 4398
   CUDA_VISIBLE_DEVICES=2 python examples/run_funsd.py  \
   --do_train --do_eval --do_predict --model_name_or_path microsoft/layoutlm-base-uncased  \
   --output_dir results/test-lm-fs-$sz-$sd --metric_for_best_model f1 --load_best_model_at_end \
   --max_steps 1000 --learning_rate 1e-5  \
    --metric_for_best_model f1 --save_strategy steps --save_steps 200 --save_strategy steps --save_total_limit 1  \
  --evaluation_strategy steps --eval_steps 200 --logging_steps 100 --logging_first_step --logging_strategy steps \
   --per_device_train_batch_size 2 --dataloader_num_workers 8  --warmup_ratio 0.1 --gradient_accumulation_steps 1  --fp16  --sz-$sz --sd-$sd --hd-$hd --heuristic-baseline

  done
  done
done
