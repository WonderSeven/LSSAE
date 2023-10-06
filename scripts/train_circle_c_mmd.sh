echo '++++++++++++++++++++++++++++++++++++++++++++++++'
echo '+++++        MMD-LSAE: Toy Circle-C        +++++'
echo '++++++++++++++++++++++++++++++++++++++++++++++++'

for seed in 0
do
  echo "Running exp with $seed"
  python main.py --gpu_ids 0 \
        --data_name ToyCircle \
        --data_path '/hdd2/qtx/Datasets/Toy_Circle/half-circle-cs-gradual.pkl' \
        --num_classes 2 \
        --data_size '[1, 2]' \
        --source-domains 15 \
        --intermediate-domains 5 \
        --target-domains 10 \
        --mode train \
        --model-func Toy_Linear_FE \
        --feature-dim 128 \
        --epochs 50 \
        --iterations 200 \
        --train_batch_size 24 \
        --eval_batch_size 50 \
        --test_epoch -1 \
        --algorithm MMD_LSAE \
        --zc-dim 20 \
        --zw-dim 20 \
        --seed $seed \
        --save_path './logs/ToyCircle_C/MMD_LSAE_18' \
        --record
  echo "=================="
done