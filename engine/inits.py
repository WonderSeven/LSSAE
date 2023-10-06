import os
import sys

sys.path.extend('../')
sys.dont_write_bytecode = True
import random
import numpy as np

import torch.nn as nn

from engine import configs
from datasets.fast_data_loader import *
from engine.hparams_registry import get_hparams
from engine.utils import create_logger, add_filehandler, format_time, split_train_test_datasets, TensorboardWriter
import network
import datasets


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_function(registry, name):
    if name in registry:
        return registry[name]
    else:
        raise Exception("{} does not support [{}], valid keys : {}".format(registry, name, list(registry.keys())))


def get_model_func(cfg):
    name = cfg.model_func.lower()
    args = {'input_shape': cfg.data_size,
            'output_dim': cfg.feature_dim}
    func = get_function(configs.Embeddings, name)
    return func(**args)


def get_cla_func(cfg):
    name = cfg.cla_func.lower()
    args = {'input_dim': cfg.feature_dim,
            'output_dim': cfg.num_classes}
    if cfg.algorithm.lower() in ['vae', 'dvae', 'diva', ]:
        args.update({'input_dim': cfg.zc_dim})
    elif cfg.algorithm.lower() == 'lssae':
        args.update({'input_dim': cfg.zc_dim + cfg.num_classes})
    elif cfg.algorithm.lower() == 'mtl':
        args.update({'input_dim': 2 * cfg.feature_dim})
    func = get_function(configs.Classifers, name)

    try:
        if issubclass(func, nn.Module):
            return func(**args)
        print('Classifier is a nn.Module object')
    except:
        print('Classifier is a func')


def get_algorithm(cfg):
    name = cfg.algorithm.lower()
    model_func = get_model_func(cfg)
    cla_func = get_cla_func(cfg)
    hparams = get_hparams(cfg)

    args = {'model_func': model_func,
            'cla_func': cla_func,
            'hparams': hparams}

    func = get_function(configs.Algorithms, name)
    return func(**args)


def get_original_dataset(cfg):
    name = cfg.data_name.lower()
    args = {'root': cfg.data_path,
            'input_shape': cfg.data_size,
            'num_classes': cfg.num_classes}
    func = get_function(configs.Datasets, name)
    return func(**args)


def get_minibatches_iterators(cfg):
    original_dataset = get_original_dataset(cfg)
    source_datasets, intermediate_datasets, target_datasets = split_train_test_datasets(original_dataset,
                                                                                      cfg.source_domains,
                                                                                      cfg.intermediate_domains,
                                                                                      cfg.target_domains)

    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=None,
        batch_size=cfg.train_batch_size,
        num_workers=cfg.num_workers)
        for i, env in enumerate(source_datasets.datasets)]

    val_loaders = [FastDataLoader(
        dataset=env,
        batch_size=cfg.eval_batch_size,
        num_workers=cfg.num_workers)
        for env in intermediate_datasets.datasets]

    test_loaders = [FastDataLoader(
        dataset=env,
        batch_size=cfg.eval_batch_size,
        num_workers=cfg.num_workers)
        for env in target_datasets.datasets]

    val_loader_names = ['Env:{}'.format(i) for i in intermediate_datasets.Environments]
    val_domain_idxes = [domain_idx for domain_idx in range(cfg.source_domains, cfg.source_domains +
                                                           cfg.intermediate_domains)]
    test_loader_names = ['Env:{}'.format(i) for i in target_datasets.Environments]
    test_domain_idxes = [domain_idx for domain_idx in range(cfg.source_domains + cfg.intermediate_domains,
                                                            cfg.source_domains + cfg.intermediate_domains + cfg.target_domains)]

    train_minibatches_iterator = zip(*train_loaders)
    val_minibatches_iterator = tuple(zip(val_loader_names, val_domain_idxes, val_loaders))
    test_minibatches_iterator = tuple(zip(test_loader_names, test_domain_idxes, test_loaders))

    return train_minibatches_iterator, val_minibatches_iterator, test_minibatches_iterator


def get_whole_dataset(cfg):
    name = 'whole'
    args = {'root': cfg.data_path}
    func = get_function(configs.Datasets, name)
    return func(**args)


def get_whole_dataloader(cfg):
    from torch.utils.data import DataLoader
    dataset = get_whole_dataset(cfg).process_dataset()
    return DataLoader(dataset, cfg.eval_batch_size, shuffle=False, num_workers=cfg.num_workers)


def get_tensorwriter(output_path, name='tfboard_files'):
    viz_path = output_path / name
    if not viz_path.exists():
        viz_path.mkdir(exist_ok=True)
    writer = TensorboardWriter(viz_path)
    return writer


def get_logger(cfg, name=None):
    logger = create_logger('SEDG')
    cur_time = format_time()
    if name is None:
        log_name = '{}_{}_{}_{}_bz{}_seed{}_{}.txt'.format(cfg.mode, cfg.data_name, cfg.model_func, cfg.cla_func,
                                                    cfg.train_batch_size, cfg.seed, cur_time)
    else:
        log_name = '{}_{}_{}_{}_bz{}_seed{}_{}.txt'.format(name, cfg.mode, cfg.data_name, cfg.model_func,
                                                    cfg.train_batch_size, cfg.seed, cur_time)

    log_path = os.path.join(cfg.save_path, log_name)
    if os.path.exists(log_path):
        os.remove(log_path)

    if cfg.record:
        # save config
        add_filehandler(logger, log_path)
    return logger
