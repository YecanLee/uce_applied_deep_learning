import torch

from uce.edit.uce import UnifiedConceptEditor


def test_model_uce():
    """Compare if the parameters of the model are stable."""
    model = UnifiedConceptEditor(
        stable_diffusion='stabilityai/stable-diffusion-2-1-base',
        lamb=0.1,
        erase_scale=1.0,
        preserve_scale=0.1,
        with_to_k=True,
        device='cuda:0' if torch.cuda.is_available() else 'cpu')
    assert model.lamb == 0.1, ('This lambda value is mismatched')
    assert model.erase_scale == 1.0, ('This erase scale value is mismatched')
    assert model.preserve_scale == 0.1, ('This preserve scale value is mismatched')
    assert model.with_to_k is True, ('This with_to_k value is mismatched')


def test_model_config_stable():
    """Compare if the model configuration file is stable between different SD Models."""
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    original_model = UnifiedConceptEditor(
        stable_diffusion='runwayml/stable-diffusion-v1-5',
        lamb=0.1,
        erase_scale=1.0,
        preserve_scale=0.1,
        with_to_k=True,
        device=device)
    state_dict, _ = original_model.state_dict()
    new_model = UnifiedConceptEditor(
        stable_diffusion='stabilityai/stable-diffusion-2-1-base',
        lamb=0.1,
        erase_scale=1.0,
        preserve_scale=0.1,
        device=device)
    state_dict_new, _ = new_model.state_dict()

    assert new_model.lamb == original_model.lamb, ('The Lambd value is mismatched')
    assert new_model.erase_scale == original_model.erase_scale, (
        'The erase scale value is mismatched')
    assert new_model.preserve_scale == original_model.preserve_scale, (
        'The preserve scale value is mismatched')
    assert new_model.with_to_k == original_model.with_to_k, (
        'The with_to_k value is mismatched')
    assert new_model.device == original_model.device, ('The device value is mismatched')
    assert state_dict.keys() == state_dict_new.keys(), (
        'The state dictionary keys are mismatched')
