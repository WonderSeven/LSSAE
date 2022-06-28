echo '++++++++++++++++++++++++++++++++++++++++++++++++'
echo '+++++         LSSAE: Toy Circle         +++++'
echo '++++++++++++++++++++++++++++++++++++++++++++++++'

for seed in 0
do
  echo "Running exp with $seed"
  python main.py --gpu_ids 3 \
        --data_name ToyCircle \
        --data_path '/data/qtx/DataSets/Toy_Circle/half-circle.pkl' \
        --num_classes 2 \
        --data_size '[1, 2]' \
        --source-domains 15 \
        --intermediate-domains 5 \
        --target-domains 10 \
        --mode train \
        --model-func Toy_Linear_FE \
        --feature-dim 512 \
        --epochs 50 \
        --iterations 200 \
        --train_batch_size 24 \
        --eval_batch_size 50 \
        --test_epoch -1 \
        --algorithm LSSAE \
        --zc-dim 20 \
        --zw-dim 20 \
        --seed $seed \
        --save_path './logs/ToyCircle/f_30' \
        --record
  echo "=================="
done