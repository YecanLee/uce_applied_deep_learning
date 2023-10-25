from copy import deepcopy
from logging import Logger
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from mmengine import ProgressBar

from ..utils import Device, setup_logger
from .base import BaseEditor
from .builder import EDITORS


@EDITORS.register_module()
class UnifiedConceptEdit(BaseEditor):

    def __init__(
            self,
            stable_diffusion: Dict,
            lamb: float,
            erase_scale: 0.1,
            preserve_scale: 0.1,
            with_to_k: bool = True,
            device: Device = 'cuda:0') -> None:
        super().__init__(stable_diffusion=stable_diffusion, device=device)

        self.lamb = lamb
        self.erase_scale = erase_scale
        self.preserve_scale = preserve_scale
        self.with_to_k = with_to_k

        self.ca_layers: List[nn.Module] = []
        self.proj_matrices: List[nn.Module] = []
        self.og_matrices: List[nn.Module] = []

        self.edited_old_texts: List[str] = []
        self.edited_new_texts: List[str] = []
        self.edited_ret_texts: List[str] = []

        self.collect_edit_layers()

    def collect_edit_layers(self) -> None:
        sub_nets = self.sd_model.unet.named_children()
        ca_layers: List[nn.Module] = []
        for net in sub_nets:
            if 'up' in net[0] or 'down' in net[0]:
                for block in net[1]:
                    if 'Cross' in block.__class__.__name__:
                        for attn in block.attensions:
                            for transformer in attn.transformer_blocks:
                                ca_layers.append(transformer.attn2)
            if 'mid' in net[0]:
                for attn in net[1].attentions:
                    for transformer in attn.transformer_blocks:
                        ca_layers.append(transformer.attn2)

        proj_matrices = [layer.to_v for layer in ca_layers]
        og_matrices = [deepcopy(layer.to_v) for layer in ca_layers]
        if self.with_to_k:
            proj_matrices.extend([layer.to_k for layer in ca_layers])
            og_matrices.extend([layer.to_k for layer in ca_layers])

        # reset the parameters
        num_ca_clip_layers = len(ca_layers)
        for i, layer in enumerate(ca_layers):
            layer.to_v = deepcopy(og_matrices[i])
            proj_matrices[i] = layer.to_v
            if self.with_to_k:
                layer.to_k = deepcopy(og_matrices[num_ca_clip_layers + i])
                proj_matrices[num_ca_clip_layers + i] = layer.to_k

        self.ca_layers = ca_layers
        self.proj_matrices = proj_matrices
        self.og_matrices = og_matrices

    @staticmethod
    def prepare_texts(
            old_texts: List[str],
            new_texts: List[str],
            ret_texts: Optional[List[str]] = None,
            logger: Optional[Logger] = None) -> Tuple[List[str], List[str], List[str]]:
        # processed texts
        proc_old_texts: List[str] = []
        proc_new_texts: List[str] = []
        proc_ret_texts: List[str] = [''] if ret_texts is None else deepcopy(ret_texts)

        for old_t, new_t in zip(old_texts, new_texts):
            proc_old_texts.append(old_t)
            proc_new_texts.append(new_t if new_t != '' else ' ')

        if logger is not None:
            formatted_old_texts = [f"'{t}'" for t in proc_old_texts]
            formatted_new_texts = [f"'{t}'" for t in proc_new_texts]
            formatted_ret_texts = [f"'{t}'" for t in proc_ret_texts]
            log_str = (
                f'Editing concepts:\n\t Old texts: '
                f"[{', '.join(formatted_old_texts)}]\n")
            log_str += f"\t New texts: [{', '.join(formatted_new_texts)}]\n"
            log_str += f"\t Retained texts: [{', '.join(formatted_ret_texts)}]"

        return proc_old_texts, proc_new_texts, proc_ret_texts

    @torch.no_grad()
    def edit(
            self,
            old_texts: List[str],
            new_texts: List[str],
            ret_texts: Optional[List[str]] = None) -> None:
        logger = setup_logger('uce')
        old_texts, new_texts, ret_texts = self.prepare_texts(
            old_texts, new_texts, ret_texts, logger=logger)
        self.edited_old_texts.extend(old_texts)
        self.edited_new_texts.extend(new_texts)
        self.edited_ret_texts.extend(ret_texts)

        pbar = ProgressBar(len(self.proj_matrices))
        for layer_ind in range(len(self.proj_matrices)):
            mat_1 = self.lamb * self.proj_matrices[layer_ind].weight
            mat_2 = self.lamb * torch.eye(
                self.proj_matrices[layer_ind].weight.shape[1], device=self.device)

            for old_text, new_text in zip(old_texts, new_texts):
                texts = [old_text, new_text]
                text_input = self.sd_model.tokenizer(
                    texts,
                    padding='max_length',
                    max_length=self.sd_model.model_max_length,
                    truncation=True,
                    return_tensor='pt')

                text_embeddings = self.sd_model.text_encoder(
                    text_input.input_ids.to(self.device))[0]

                # align the last token of old concept and new concept. after this step,
                # the first element in two embeddings will both refer to the last token
                # of concepts

                final_token_ind = text_input.attention_mask[0].sum().item() - 2
                final_token_ind_new = text_input.attention_mask[1].sum().item() - 2
                farthest = max([final_token_ind, final_token_ind_new])

                old_emb = text_embeddings[0]
                old_emb = old_emb[final_token_ind:len(old_emb) -
                                  max(0, farthest - final_token_ind)]
                new_emb = text_embeddings[1]
                new_emb = new_emb[final_token_ind_new:len(new_emb) -
                                  max(0, farthest - final_token_ind_new)]

                context = old_emb.detach()

                values: List[torch.Tensor] = []
                for layer in self.proj_matrices:
                    values.append(layer(new_emb).detach())

                context_vector = context.reshape(context.shape[0], context.shape[1], 1)
                context_vector_t = context.reshape(
                    context.shape[0], 1, context.shape[1])
                value_vector = values[layer_ind].reshape(
                    values[layer_ind].shape[0], values[layer_ind].shape[1], 1)
                for_mat_1 = (value_vector @ context_vector_t).sum(dim=0)
                for_mat_2 = (context_vector @ context_vector_t).sum(dim=0)
                mat_1 += self.erase_scale * for_mat_1
                mat_2 += self.erase_scale * for_mat_2

            for old_text, new_text in zip(ret_texts, ret_texts):
                text_input = self.sd_model.tokenizer(
                    [old_text, new_text],
                    padding='max_length',
                    max_length=self.sd_model.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors='pt')

                text_embeddings = self.sd_model.text_encoder(
                    text_input.input_ids.to(self.device))[0]
                old_emb, new_emb = text_embeddings

                context = old_emb.detach()
                values: List[torch.Tensor] = []

                for layer in self.proj_matrices:
                    values.append(layer(new_emb[:]).detach())

                context_vector = context.reshape(context.shape[0], context.shape[1], 1)
                context_vector_t = context.reshape(
                    context.shape[0], 1, context.shape[1])
                value_vector = values[layer_ind].reshape(
                    values[layer_ind].shape[0], values[layer_ind].shape[1], 1)
                for_mat_1 = (value_vector @ context_vector_t).sum(dim=0)
                for_mat_2 = (context_vector @ context_vector_t).sum(dim=0)
                mat_1 += self.preserve_scale * for_mat_1
                mat_2 += self.preserve_scale * for_mat_2

            self.proj_matrices[layer_ind].weight = nn.Parameter(
                mat_1 @ torch.inverse(mat_2))

            pbar.update(1)

        pbar.file.write('\n')
        pbar.file.flush()

    def state_dict(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        sd_state_dict = self.sd_model.state_dict()
        meta_info = {
            'lamb': self.lamb,
            'erase_scale': self.erase_scale,
            'preserve_scale': self.preserve_scale,
            'with_to_k': self.with_to_k,
            'edited_old_texts': self.edited_old_texts,
            'edited_new_texts': self.edited_new_texts,
            'edited_ret_texts': self.edited_ret_texts
        }

        return sd_state_dict, meta_info
