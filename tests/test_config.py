# read the config file and store it in a dictionary


def load_config(path):
    """Read the config file and store it in a dictionary for later usage."""
    config = {}
    with open(path) as f:
        exec(f.read(), config)
    return config


def test_editor_config_file():
    """Compare model configuration files between different SD Models."""
    config_file_path = 'configs/edit/sd_21_uce.py'
    config_file = load_config(config_file_path)
    assert 'editor' in config_file, (
        'the config file has no configuration parameters for the editor')
    assert config_file['editor']['type'] == 'UnifiedConceptEditor', (
        'This Editor type is not supported yet')
    assert config_file['editor']['stable_diffusion'] == (
        'stabilityai/stable-diffusion-2-1-base'), (
            'This stable diffusion model is not supported yet')
    assert config_file['editor']['lamb'] == 0.5, ('This lambda value is mismatched')
    assert config_file['editor']['erase_scale'] == 1.0, (
        'This erase scale value is mismatched')
    assert config_file['editor']['preserve_scale'] == 0.1, (
        'This preserve scale value is mismatched')
    assert config_file['editor']['with_to_k'], ('This with_to_k value is mismatched')