import os.path as osp
import time
from argparse import ArgumentParser

import torch
from mmengine import Config, DictAction, mkdir_or_exist
from torch.utils.data import DataLoader

from uce.datasets import DATASETS
from uce.iba import IBA_COMPONENTS
from uce.utils import setup_logger


def parse_args():
    parser = ArgumentParser('Train IBA on StableDiffusion.')

    parser.add_argument('config', help='Path to config file.')
    parser.add_argument('--ckpt', '-c', help='Checkpoint of the estimator of IBA')
    parser.add_argument(
        '--estimate-only', '-e', action='store_true', help='Only perform estimation.')
    parser.add_argument(
        '--work-dir', '-w', help='Working directory to save the output files.')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU ID.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='key value pairs to override the config file entries. The pairs need to '
        'be in format of xxx=yyy')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    estimate_only = args.estimate_only
    work_dir = args.work_dir
    mkdir_or_exist(work_dir)
    device = f'cuda:{args.gpu_id}'

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
    log_file = osp.join(work_dir, f'{timestamp}.log')
    logger = setup_logger('uce', filepath=log_file)

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg.dump(osp.join(work_dir, osp.basename(cfg.filename)))
    logger.info(f"Using config:\n{'=' * 60}\n{cfg.pretty_text}\n{'=' * 60}\n")

    iba_runner = IBA_COMPONENTS.build(
        cfg.runner, default_args={
            'device': device, 'estimator_ckpt': args.ckpt
        })

    if args.ckpt is None:
        est_dataset = DATASETS.build(cfg['data']['estimation'])
        est_collate_fn = est_dataset.get_collate_fn()
        est_data_loader = DataLoader(
            est_dataset,
            collate_fn=est_collate_fn,
            **cfg['data']['data_loader']['estimation'])

        iba_runner.run_estimation(est_data_loader)

    if not estimate_only:
        attr_dataset = DATASETS.build(cfg['data']['attribution'])
        attr_collate_fn = attr_dataset.get_collate_fn()
        attr_data_loader = DataLoader(
            attr_dataset,
            collate_fn=attr_collate_fn,
            **cfg['data']['data_loader']['attribution'])
        iba_runner.run_analysis(attr_data_loader, work_dir=work_dir)
    else:
        state_dict = iba_runner.iba.estimator.state_dict()
        save_ckpt_path = osp.join(work_dir, 'iba_estimator.pt')
        torch.save(state_dict, save_ckpt_path)
        logger.info(f'IBA estimator checkpoint is saved to: {save_ckpt_path}')


if __name__ == '__main__':
    main()
