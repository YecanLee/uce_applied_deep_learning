generator = dict(
    type='ClassifierFreeGenerator',
    prompts_path='data/prompts/imagenet_prompts.csv',
    base='1.4',
    guidance_scale=7.5,
    img_size=(512, 512),
    ddim_steps=100,
    num_samples=1,
    from_case=0,
    till_case=100000,
)
