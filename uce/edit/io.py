import os.path as osp

import torch
from mmengine import dump

from ..utils import setup_logger
from .base import BaseEditor


def load_editor(editor: BaseEditor, ckpt_path: str) -> BaseEditor:
    logger = setup_logger('uce')
    logger.info(f'Loading checkpoint: {ckpt_path}')

    state_dict = torch.load(ckpt_path, map_location=editor.device)
    editor.sd_model.load_state_dict(state_dict)

    id_code = osp.splitext(osp.basename(ckpt_path))[0].split('_')[-1]
    logger.info(
        f"The checkpoint has id code: '{id_code}'. "
        f'This code will be set for the editor.')
    editor.id_code = id_code

    return editor


def save_editor(editor: BaseEditor, ckpt_path: str) -> None:
    logger = setup_logger('uce')
    state_dict, meta_info = editor.state_dict()
    torch.save(state_dict, ckpt_path)
    logger.info(f'Checkpoint is saved to: {ckpt_path}')

    meta_info_path = osp.splitext(ckpt_path)[0] + '.json'
    dump(meta_info, meta_info_path)
    logger.info(f'Meta information dictionary is saved to: {meta_info_path}')
