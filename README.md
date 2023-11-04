# Unified Concept Editing in Diffusion Models
This is a re-implementation of the paper _Unified Concept Editing in Diffusion Models, Robit Gandikota et. al._

## Installation
Install your package in editable mode via `pip install -e .` (do no forget the ".").

## Optional (For developers)
Install pre-commit hooks via `pre-commit install`.

## Commands
To generate images (using the pre-trained diffusion models) for prompts from index 0
to index 10, and generate 10 samples for each prompt.
```shell
python tools/generate_images.py \
  configs/generate/sd_14_ori_imagenet.py \
  -w workdirs/debug/ \
  --cfg-options generator.till_case=10 generator.num_samples=10
```
