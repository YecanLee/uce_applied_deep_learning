import os.path as osp
from typing import List, Optional, Tuple

import numpy as np
import torch
from mmengine import dump, load

from ..utils import setup_logger
from .base import BaseEditor


def load_editor(editor: BaseEditor, ckpt_path: str, meta_info_path: str) -> BaseEditor:
    logger = setup_logger('uce')
    logger.info(f'Loading checkpoint: {ckpt_path}')

    state_dict = torch.load(ckpt_path, map_location=editor.device)
    meta_info = load(meta_info_path)
    editor.load_state_dict(state_dict, meta_info)

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

    meta_info_path = osp.splitext(ckpt_path)[0] + '_meta.json'
    dump(meta_info, meta_info_path)
    logger.info(f'Meta information dictionary is saved to: {meta_info_path}')


def parse_concepts(
        edit_concepts: str,
        guided_concepts: Optional[str] = None,
        preserve_concepts: Optional[str] = None,
        num_preserve_concepts: Optional[int] = None,
        with_extra: Optional[str] = None) -> Tuple[List[str], List[str], List[str]]:
    logger = setup_logger('uce')

    if osp.exists(edit_concepts):
        logger.info(f'Loading editing concepts from: {edit_concepts}')
        edit_concepts = np.loadtxt(edit_concepts, dtype=str)
    else:
        edit_concepts = [c.strip() for c in edit_concepts.split(',')]

    if with_extra is not None:
        if with_extra == 'artist':
            extra_prompts = [
                'painting by {concept}',
                'art by {concept}',
                'artwork by {concept}',
                'picture by {concept}',
                'style by {concept}'
            ]
        elif with_extra == 'object':
            extra_prompts = [
                'image of {concept}',
                'photo of {concept}',
                'portrait of {concept}',
                'picture of {concept}',
                'painting of {concept}',
            ]
        else:
            raise ValueError(
                f"with_extra can only be 'artist' or 'object' or None. "
                f'But got {with_extra}')
    else:
        extra_prompts = []

    extra_length = len(extra_prompts)
    parsed_edit_concepts = []
    for concept in edit_concepts:
        parsed_edit_concepts.append(concept)
        parsed_edit_concepts.extend([p.format(concept) for p in extra_prompts])

    if guided_concepts is not None:
        if osp.exists(guided_concepts):
            logger.info(f'Loading guided concepts from: {guided_concepts}')
            guided_concepts = np.loadtxt(guided_concepts, dtype=str)
        else:
            guided_concepts = [c.strip() for c in guided_concepts.split(',')]

        if len(guided_concepts) == 1:
            parsed_guided_concepts = [guided_concepts[0] for _ in parsed_edit_concepts]
        else:
            parsed_guided_concepts = [
                c for _ in range(extra_length + 1) for c in guided_concepts
            ]
    else:
        parsed_guided_concepts = [' ' for _ in parsed_edit_concepts]

    assert len(parsed_edit_concepts) == len(parsed_guided_concepts)

    if preserve_concepts is not None:
        if osp.exists(preserve_concepts):
            logger.info(f'Loading preserve concepts from: {preserve_concepts}')
            preserve_concepts = np.loadtxt(preserve_concepts, dtype=str)
        else:
            preserve_concepts = [c.strip() for c in preserve_concepts.split(',')]
        # filter all the concepts that are already in the edit_concepts
        lower_edit_concepts = [c.lower() for c in parsed_edit_concepts]
        preserve_concepts = [
            c for c in preserve_concepts if c.lower() not in lower_edit_concepts
        ]
        if num_preserve_concepts is not None:
            preserve_concepts = np.random.choice(
                preserve_concepts, num_preserve_concepts, replace=False)
    else:
        preserve_concepts = []
    parsed_preserved_concepts = [''] + preserve_concepts

    return parsed_edit_concepts, parsed_guided_concepts, parsed_preserved_concepts
