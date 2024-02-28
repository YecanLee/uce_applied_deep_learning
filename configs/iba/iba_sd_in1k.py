data = dict(
    estimation=dict(
        type='IBAPromptsDataset', csv_path='data/prompts/imagenet_prompts.csv'),
    attribution=dict(
        type='IBAPromptsDataset',
        csv_path='data/prompts/iba_imagenet_prompts.csv',
        case_range=[0, 2]),
    data_loader=dict(
        estimation=dict(
            batch_size=4,
            shuffle=True,
            num_workers=8,
        ),
        attribution=dict(
            batch_size=1,
            shuffle=False,
            num_workers=2,
        )),
)

runner = dict(
    type='IBARunner',
    iba=dict(
        type='InformationBottleneck',
        init_alpha_val=5.0,
        threshold=0.01,
        min_noise_std=0.01),
    estimation_cfg=dict(num_samples=5000),
    analysis_cfg=dict(
        min_num_samples=1000, info_loss_weight=10.0, lr=1.0, batch_size=1),
    inference_cfg=dict(
        guidance_scale=7.5,
        height=512,
        width=512,
        num_inference_steps=50,
        num_images_per_prompt=10,
    ),
    classifier=dict(model_name='maxvit_tiny_tf_512.in1k', pretrained=True),
    stable_diffusion='stabilityai/stable-diffusion-2-1-base',
)
