echo '++++++++++++++++++++++++++++++++++++++++++++++++'
echo '+++++       MMD_LSAE: Rotated MNIST        +++++'
echo '++++++++++++++++++++++++++++++++++++++++++++++++'

for seed in 0
do
  echo "Running exp with $seed"
  python main.py --gpu_ids 0 \
        --data_name RMNIST \
        --data_path '/hdd2/qtx/Datasets'\
        --num_classes 10 \
        --data_size '[1, 28, 28]' \
        --source-domains 10 \
        --intermediate-domains 3 \
        --target-domains 6 \
        --mode train \
        --model-func MNIST_CNN \
        --feature-dim 128 \
        --epochs 80 \
        --iterations 200 \
        --train_batch_size 48 \
        --eval_batch_size 48 \
        --test_epoch -1 \
        --algorithm MMD_LSAE \
        --zc-dim 32 \
        --zw-dim 32 \
        --seed $seed \
        --save_path './logs/RMNIST/MMD_LSAE' \
        --record
  echo "=================="
done