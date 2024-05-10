# conda activate layoutlmft
path='/home/prashant/unilm/layoutlmv3'
deg=4
theta=60

for sz in 7
do
 for sd in 2
  do
  for scale in 4
   do
     cd $path
     dir_path=test-layoutlmv3-cord-gat-closest-$sz-$sd-h4-d4
     echo "===== CORD FS - scale-$scale sz-$sz-sd-$sd ===="

     CUDA_VISIBLE_DEVICES=0 python examples/test_cord.py --model_name_or_path microsoft/layoutlmv3-base \
     --output_dir results/$dir_path-scale-$scale-v2 \
      --do_predict --logging_steps 100 --logging_first_step --logging_strategy steps \
      --save_total_limit 1 --max_steps 1000 --warmup_ratio 0.1 \
     --fp16 --dir_$dir_path --sz-$sz --sd-$sd --scale-$scale
  done
  done
done