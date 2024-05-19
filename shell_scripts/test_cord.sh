# conda activate layoutlmft
path=$(pwd)
echo "Current working directory : $path"
heuristic='nearest' # Heuristics available: nearest, angles, baseline
deg=4 # Degree of the graph
hd=4 # Number of attention heads in the GAT
manip_type='shift' # Type of image manipulations. Valid inputs: shift, rotate and scale
# Run a loop for different few-shot sizes and seeds
for sz in 5
do
 for sd in 2
  do
  for manip_param in 10
   do
     cd $path
     dir_path=test-layoutlmv3-cord-gat-closest-$sz-$sd-h4-d4
     echo "===== CORD - $manip_type-$manip_param- sz-$sz-sd-$sd ===="

     CUDA_VISIBLE_DEVICES=0 python examples/test_cord.py --model_name_or_path microsoft/layoutlmv3-base \
     --output_dir results/$dir_path-$manip_type-$manip_param \
      --do_predict --logging_steps 100 --logging_first_step --logging_strategy steps \
      --save_total_limit 1 --max_steps 1000 --warmup_ratio 0.1 \
     --fp16 --dir-$dir_path --sz-$sz --sd-$sd --heuristic-$heuristic --manip_type-$manip_type --manip_param-$manip_param $path/data/path_config.json
  done
  done
done