# Unified Concept Editing in Diffusion Models
This is a re-implementation of the paper _Unified Concept Editing in Diffusion Models, Robit Gandikota et. al._

## Installation
Install your package in editable mode via `pip install -e .` (do not forget the ".").

## Optional (For developers)
Install pre-commit hooks via `pre-commit install`.

## Commands

### Concept Erasing
To erase multiple concepts e.g. "car,bicycle,bus". Please refer to
`tools/edit_model.py` for detailed usage hints.

```shell
python tools/edit_model.py \
  configs/edit/sd_14_uce.py \
  "car,bicyle,bus"
  -w workdirs/debug/
```




### Image Generation
To generate images (using the **pre-trained** diffusion models) for prompts from index 0
to index 10, and generate 10 samples for each prompt. See `tools/generate_images.py`
for detailed usage hints.
```shell
python tools/generate_images.py \
  configs/generate/sd_14_cars.py \
  -w workdirs/debug/ \
  --cfg-options generator.till_case=10 generator.inference_cfg.num_images_per_prompt=10
```

To generate images (using **edited** diffusion models). You need to specify the edited
model's weights by adding `-c path/to/checkpoint.pt`.

To use your own prompts, you need to prepare a csv file that has the same format as
`data/prompts/cars_prompts.csv`. Then, you need to specify the path to this
prompt csv the config: `generator.prompts_path=path/to/your/file.csv`. Or you modify it
in terminal by using `--cfg-options generator.prompts_path=path/to/your/file.csv`.
