from models.discriminators import MultiBandSTFTDiscriminator, SSDiscriminatorBlock

def prepare_discriminator(config):
    
    # disc_type = config['discriminator']['type']
    disc_config = config['discriminator']
    mbstftd_config = disc_config.get('MultiBandSTFTDiscriminator_config', None)
    mpd_config = disc_config.get('PeriodDiscriminator_config', None)
    
    if not mbstftd_config and not mpd_config:
        raise ValueError(f"At least one discriminator is required")

    discriminator = SSDiscriminatorBlock(
        # STFTD config
        sd_num=len(mbstftd_config['n_fft_list']) if mbstftd_config else 0,
        C=mbstftd_config['C'] if mbstftd_config else None,
        n_fft_list=mbstftd_config['n_fft_list'] if mbstftd_config else [],
        hop_len_list=mbstftd_config['hop_len_list'] if mbstftd_config else [],
        band_split_ratio=mbstftd_config['band_split_ratio'] if mbstftd_config else [],
        sd_mode='BS',
        
        # MPD config
        pd_num=len(mpd_config['period_list']) if mpd_config else 0,
        period_list=mpd_config['period_list'] if mpd_config else [],
        C_period=mpd_config['C_period'] if mpd_config else None,
    )

    # Print information about the loaded model
    # Print information about the discriminator
    print("########################################")
    print("Discriminator Configurations:")
    if mbstftd_config:
        print(f"- STFTD Config: {mbstftd_config}")
    else:
        print("- STFTD is not used.")
    
    if mpd_config:
        print(f"- MPD Config: {mpd_config}")
    else:
        print("- MPD is not used.")
    
    print(f"Discriminator Parameters: {sum(p.numel() for p in discriminator.parameters() if p.requires_grad) / 1_000_000:.2f}M")
    print("########################################")

    return discriminator

def prepare_generator(config, MODEL_MAP):
    gen_type = config['generator']['type']
    
    if gen_type not in MODEL_MAP:
        raise ValueError(f"Unsupported generator type: {gen_type}")
    
    ModelClass = MODEL_MAP[gen_type]
    
    # Retrieve the parameters for the generator from the config
    model_params = {k: v for k, v in config['generator'].items() if k not in ['type']}
    rvq_config = config['generator'].get('rvq_config', None)
    
    # Print information about the loaded model
    print("########################################")
    print(f"Instantiating {gen_type} Generator with parameters:")
    for key, value in model_params.items():
        print(f"  {key}: {value}")
    if rvq_config:
        print("  rvq_config:")
        for key, value in rvq_config.items():
            print(f"    {key}: {value}")
    print(f"  type: {gen_type}")
    generator = ModelClass(**model_params)
    print(f"Generator Parameters: {sum(p.numel() for p in generator.parameters() if p.requires_grad) / 1_000_000:.2f}M")
    print("########################################")
    
    return generator