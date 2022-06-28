# -*- coding:utf-8 -*-
"""
Created by 'tiexin'
"""

import argparse
import configs


def parse_args():
    # Parse command line / default arguments
    parser = argparse.ArgumentParser(description='LSSAE Experiments')
    # Data parameters
    parser.add_argument('--data_name', type=str, default='RMNIST', help='Dataset to use')
    parser.add_argument('--data_path', type=str, default=configs.mnist_path,
                        help='Path, where datasets are stored')
    parser.add_argument('--num_classes', type=int, default=2, help='The number of classes in dataset')
    parser.add_argument('--data_size', default=[1, 28, 28], help='Each sample size in dataset')
    parser.add_argument('--source-domains', type=int, default=10)
    parser.add_argument('--intermediate-domains', type=int, default=3)
    parser.add_argument('--target-domains', type=int, default=6)

    # Training parameters
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model-func', type=str, default='MNIST_CNN', help='Backbone architecture')
    parser.add_argument('--feature-dim', type=int, default=128, help='The dims fo model_func output')
    parser.add_argument('--cla-func', type=str, default='Linear_Cla')
    parser.add_argument('--algorithm', type=str, default='ERM')

    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--train_batch_size', type=int, default=48,
                        help='Batch size for ProtoTransfer')
    parser.add_argument('--eval_batch_size', type=int, default=48,
                        help='Batch size for ProtoTransfer')
    parser.add_argument('--test_epoch', type=int, default=-1,
                        help='Epoch when model test')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='The number of workers for data loaders')
    # Parameters for LSSAE
    parser.add_argument('--zc-dim', type=int, default=20, help='The dims for latent variable')
    parser.add_argument('--zw-dim', type=int, default=20, help='The dims for env latent variable')

    # Saving and loading parameters
    parser.add_argument('--save_path', type=str, default=configs.save_dir, help='Save path')
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--record', action='store_true', default=False,
                        help='Whether to record the model training procedure')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--gpu_ids', type=str, default='0')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    from engine import trainer

    args = parse_args()
    trainer = trainer.Trainer(args)
    trainer.train()
