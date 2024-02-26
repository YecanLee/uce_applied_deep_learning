import pytest
from uce.edit.uce import UnifiedConceptEditor

def test_model_uce():
    model = UnifiedConceptEditor(
        type='UnifiedConceptEditor',
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



