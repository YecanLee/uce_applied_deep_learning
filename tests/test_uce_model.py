import pytest
from uce.edit.uce import UnifiedConceptEditor

def test_model_uce():
    model = UnifiedConceptEditor(
        stable_diffusion='stabilityai/stable-diffusion-2-1-base',
        lamb=0.1,
        erase_scale=1.0,
        preserve_scale=0.1,
        with_to_k=True,
        device='cuda:0'
    )
    assert model.lamb == 0.1, "This lambda value is mismatched"
    assert model.erase_scale == 1.0, "This erase scale value is mismatched"
    assert model.preserve_scale == 0.1, "This preserve scale value is mismatched"
    assert model.with_to_k is True, "This with_to_k value is mismatched"
    assert isinstance(model.device, str) and model.device == "cuda:0", "You are not using the GPU, please enable it"

def test_layer_collection():
    model = UnifiedConceptEditor(
        "some_repo", 0.1, 1.0, 0.1, True, 'cuda:0'
    )
    model.collect_edit_layers()  
    assert len(model.ca_layers) > 0  


def test_model_config_stable():
    """
    Compare if the model configuration file is stable between different Stable Diffusion Models
    """
    original_model = UnifiedConceptEditor(
        stable_diffusion='stabilityai/stable-diffusion-1-5-base',
        lamb=0.1,
        erase_scale=1.0,
        preserve_scale=0.1,
        with_to_k=True,
        device='cuda:0'
    )
    state_dict= original_model.state_dict()
    new_model = UnifiedConceptEditor(
        stable_diffusion='stabilityai/stable-diffusion-2-1-base',
        lamb=0.1,
        erase_scale=1.0,
        preserve_scale=0.1,
        with_to_k=True,
        device='cuda:0'
    )
    new_model.load_state_dict(state_dict)

    assert new_model.lamb == original_model.lamb, "The Lambd value is mismatched"
    assert new_model.erase_scale == original_model.erase_scale, "The erase scale value is mismatched"
    assert new_model.preserve_scale == original_model.preserve_scale, "The preserve scale value is mismatched"
    assert new_model.with_to_k == original_model.with_to_k, "The with_to_k value is mismatched"
    assert new_model.device == original_model.device, "The device value is mismatched, you need to put every model to the same device to compare the performance of the models"

