generator = dict(
    type='StableDiffusionGenerator',
    prompts_path='data/prompts/cars_prompts.csv',
    stable_diffusion='CompVis/stable-diffusion-v1-4',
    inference_cfg=dict(
        guidance_scale=7.5,
        height=512,
        width=512,
        num_inference_steps=100,
        num_images_per_prompt=1,
    ),
    from_case=0,
    till_case=1000000,
)
