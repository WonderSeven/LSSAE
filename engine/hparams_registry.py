# https://github.com/facebookresearch/DomainBed/blob/main/domainbed/hparams_registry.py
# The specific hyper-parameters for each algorithm

import numpy as np
from engine.utils import misc


def _define_hparam(hparams, hparam_name, default_val, random_val_fn):
    hparams[hparam_name] = (hparams, hparam_name, default_val, random_val_fn)


def _hparams(cfg):
    """
    Global registry of hyperparams. Each entry is a (default, random) tuple.
    New algorithms.py / networks / etc. should add entries here.
    """
    SMALL_IMAGES = ['RMNIST']

    Digit_Data = ['ToyCircle', 'ToySine', 'CoverType', 'PowerSupply']

    random_seed = cfg.seed
    dataset = cfg.data_name
    algorithm = cfg.algorithm

    hparams = {}

    def _hparam(name, default_val, random_val_fn):
        """Define a hyperparameter. random_val_fn takes a RandomState and
        returns a random hyperparameter value."""
        assert (name not in hparams)
        random_state = np.random.RandomState(
            misc.seed_hash(random_seed, name)
        )
        hparams[name] = (default_val, random_val_fn(random_state))

    # Define global parameters
    # Note that domain_num is for test only
    _hparam('source_domains', cfg.source_domains, lambda r: cfg.source_domains)
    _hparam('num_classes', cfg.num_classes, lambda r: cfg.num_classes)
    _hparam('data_size', cfg.data_size, lambda r: cfg.data_size)

    if dataset in SMALL_IMAGES:
        _hparam('lr', 1e-3, lambda r: 10 ** r.uniform(-4.5, -2.5))
        _hparam('weight_decay', 0., lambda r: 0.)
    elif dataset in Digit_Data:
        _hparam('lr', 5e-5, lambda r: 10 ** r.uniform(-5, -3.5))
        _hparam('weight_decay', 0., lambda r: 10 ** r.uniform(-6, -2))
    else:
        _hparam('lr', 5e-5, lambda r: 10 ** r.uniform(-5, -3.5))
        _hparam('weight_decay', 0., lambda r: 10 ** r.uniform(-6, -2))

    if algorithm in ['VAE', 'DIVA', 'LSSAE']:
        _hparam('zc_dim', cfg.zc_dim, lambda r: cfg.zc_dim)
        _hparam('stochastic', True, lambda r: True)
        # Params for DIVA only
        _hparam('zdy_dim', cfg.zw_dim, lambda r: cfg.zw_dim)
        # Params for LSSAE only
        _hparam('zw_dim', cfg.zw_dim, lambda r: cfg.zw_dim)
        _hparam('zv_dim', cfg.num_classes, lambda r: cfg.num_classes)
        _hparam('coeff_y', 80, lambda r: 80)
        _hparam('coeff_ts', 20, lambda r: 20)

    return hparams


def get_hparams(cfg):
    return {a: b for a, (b, c) in _hparams(cfg).items()}
