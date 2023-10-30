import os.path as osp
import time
from argparse import ArgumentParser

import torch
from mmengine import Config, DictAction, mkdir_or_exist

from uce.generate import GENERATORS
from uce.utils import setup_logger


def parse_args():
    parser = ArgumentParser('Generate images.')

    parser.add_argument('config', help='Path to config file.')
    parser.add_argument(
        '--editor-ckpt',
        '-c',
        help='Path to the editor checkpoint file. If None, no checkpoint will be '
        'loaded. The default unedited Stable Diffusion weights will be used.')
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
    work_dir = args.work_dir
    mkdir_or_exist(work_dir)
    imgs_save_path = osp.join(work_dir, 'images')
    mkdir_or_exist(imgs_save_path)

    device = torch.device(f'cuda:{args.gpu_id}')

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
    log_file = osp.join(work_dir, f'{timestamp}.log')
    logger = setup_logger('uce', filepath=log_file)

    cfg = Config.fromfile(args.config)
    cfg.generator.update({'device': device})
    logger.info(f"Using config:\n{'=' * 60}\n{cfg.pretty_text}\n{'=' * 60}\n")

    generator = GENERATORS.build(cfg.generator)
    generator.load_state_dict(args.editor_ckpt)
    generator.generate(out_path=imgs_save_path)
    logger.info(f'Images are saved to: {imgs_save_path}')


if __name__ == '__main__':
    main()
