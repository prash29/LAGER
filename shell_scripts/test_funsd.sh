# conda activate layoutlmft
# path='/home/prashant/unilm/layoutlmv3'
path='/home/prashant/lager_repo'
deg=4
theta=60
heuristic='nearest'
manip_type='shift'
for sz in 4
do
 for sd in 0
  do
  for manip_param in 10
   do
     cd $path
     dir_path=test-layoutlmv3-gat-angles-full
     echo "===== FUNSD FS - $manip_type-$manip_param- sz-$sz-sd-$sd ===="

     CUDA_VISIBLE_DEVICES=0 python examples/test_funsd.py --model_name_or_path microsoft/layoutlmv3-base \
     --output_dir results/$dir_path-$manip_type-$manip_param \
      --do_predict --logging_steps 100 --logging_first_step --logging_strategy steps \
      --save_total_limit 1 --max_steps 1000 --warmup_ratio 0.1 \
     --fp16 --dir_$dir_path --sz-$sz --sd-$sd --heuristic-$heuristic --manip_type-$manip_type --manip_param-$manip_param /home/prashant/lager_repo/data/path_config.json
  done
  done
done
