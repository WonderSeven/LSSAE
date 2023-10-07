echo '++++++++++++++++++++++++++++++++++++++++++++++++'
echo '++++++          LSSAE: PowerSupply        ++++++'
echo '++++++++++++++++++++++++++++++++++++++++++++++++'

for seed in 0
do
  echo "Running exp with $seed"
  python main.py --gpu_ids 0 \
        --data_name PowerSupply \
        --data_path '/hdd2/qtx/Datasets/PowerSupply/powersupply.arff' \
        --num_classes 2 \
        --image_size '[1, 2]' \
        --source-domains 15 \
        --intermediate-domains 5 \
        --target-domains 10 \
        --mode train \
        --epochs 50 \
        --iterations 200 \
        --train_batch_size 48 \
        --eval_batch_size 24 \
        --test_epoch -1 \
        --algorithm LSSAE \
        --zc-dim 20 \
        --zw-dim 20 \
        --seed $seed \
        --save_path './logs/PowerSupply' \
        --record
  echo "=================="
done