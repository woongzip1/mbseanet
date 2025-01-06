from models.discriminators import MultiBandSTFTDiscriminator, SSDiscriminatorBlock

def prepare_discriminator(config):
    
    disc_type = config['discriminator']['type']

    if disc_type == "MultiBandSTFTDiscriminator":
        disc_config = config["discriminator"]['MultiBandSTFTDiscriminator_config']
        discriminator = SSDiscriminatorBlock(
            sd_num=len(disc_config['n_fft_list']),
            C=disc_config['C'],
            n_fft_list=disc_config['n_fft_list'],
            hop_len_list=disc_config['hop_len_list'],
            sd_mode='BS',
            band_split_ratio=disc_config['band_split_ratio']
        )
    else:
        raise ValueError(f"Unsupported discriminator type: {disc_type}")

    # Print information about the loaded model
    print("########################################")
    print(f"Discriminator Type: {disc_type}")
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
    
    # Print information about the loaded model
    print("########################################")
    print(f"Instantiating {gen_type} Generator with parameters:")
    for key, value in model_params.items():
        print(f"  {key}: {value}")
    print(f"  type: {gen_type}")
    generator = ModelClass(**model_params)
    print(f"Generator Parameters: {sum(p.numel() for p in generator.parameters() if p.requires_grad) / 1_000_000:.2f}M")
    print("########################################")
    
    return generator