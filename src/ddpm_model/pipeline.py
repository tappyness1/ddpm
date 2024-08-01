import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler

WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

# uncond_prompt also known as negative prompt where you don't want it to generate something

def generate_images(prompt: str, 
                    uncond_prompt: str, 
                    input_image=None, 
                    strength=0.8, # how much to pay attention to the image when generating the image. The higher the strength the more noise we add, hence more "creative" the image will be generated
                    do_cfg = True, # if we want to do classifier-free guidance not config!
                    cfg_scale = 7.5, 
                    sampler_name = 'ddpm', 
                    n_inference_steps=50, 
                    models={}, 
                    seed = None, 
                    device = None, 
                    idle_device=None, 
                    tokenizer = None):
    
    with torch.no_grad():
        if not (0 < strength <= 1):
            raise ValueError('strength must be in (0, 1]')
    
    # move to cpu. Useful if we want to store something if we don't need it until later
    if idle_device: 
        to_idle: lambda x: x.to(idle_device)
    else:
        to_idle: lambda x: x

    generator = torch.Generator(device = device)
    if seed is None: 
        generator.seed()
    else: 
        generator.manual_seed(seed)

    clip = models["clip"]
    clip.to(device)

    # in classifier-free guidance, output = w * (input_condprompt - input_uncondprompt) + input_uncondprompt per the paper
    if do_cfg: # want to do classifier-free 
        cond_tokens = tokenizer.batch_encode_plus([prompt], paddings = "max_length", max_length = 77).input_ids
        cond_tokens = torch.tensor(cond_tokens, dtype = torch.long, device = device) # (Batch_size, Seq_len)
        cond_context = clip(cond_tokens) # get the embeddings, (Batch_size, Seq_len) -> (Batch_size, Seq_len, Embed_dim)

        uncond_tokens = tokenizer.batch_encode_plus([uncond_prompt], paddings = "max_length", max_length = 77).input_ids
        uncond_tokens = torch.tensor(uncond_tokens, dtype = torch.long, device = device)
        uncond_context = clip(uncond_tokens)

        context = torch.cat([cond_context, uncond_context]) # 2 x  Seq_len x Embed_dim (not 2 x Batch_size x Seq_len x Embed_dim????)

    else: 
        tokens = tokenizer.batch_encode_plus([prompt], paddings = "max_length", max_length = 77).input_ids
        tokens = torch.tensor(tokens, dtype = torch.long, device = device)
        context = clip(tokens) # 1 x Seq_len x embed_dim

    to_idle(clip)

    # since we are only doing DDPM, we don't necessarily need to check for other samplers
    if sampler_name == 'ddpm':
        sampler = DDPMSampler(generator)
        sampler.set_inference_steps(n_inference_steps)
    
    else:
        raise ValueError(f'Unknown sampler: {sampler_name}') 
    
    latents_shape = (1,4, LATENTS_HEIGHT, LATENTS_WIDTH)

    if input_image: 
        encoder = models['encoder']
        encoder.to(device)

        input_image_tensor = input_image.resize((WIDTH, HEIGHT))
        input_image_tensor = np.array(input_image_tensor)
        # height x width x channels
        input_image_tensor = torch.tensor(input_image_tensor, dtype = torch.float32)

        # unet needs each pixel to be in the [-1,1] range so we will convert to that

        input_image_tensor = rescale(input_image_tensor, (0,255), (-1,1))

        # add the batch dimension and swap the channels
        input_image_tensor = input_image.unsqueeze(0).permute(0,3,1,2)

        encoder_noise = torch.randn(latents_shape, generator = generator, device = device)
        
        # run the image through the encoder of the VAE
        latents = encoder(input_image_tensor, encoder_noise)

        # add noise to the image. The more noise there is, the more we have to remove
        sampler.set_strength(strength = strength)
        latents = sampler.add_noise(latents, sampler.timesteps[0])

        to_idle(encoder)
        
    else: 
        # start with random noise since we don't have any image fed in. Noise is N(0,1)
        latents = torch.randn(latents_shape, generator = generator, device = device)

    diffusion = models['diffusion']
    diffusion.to(device)

    timesteps = tqdm(sampler.timesteps)
    for i, timestep in enumerate(timesteps):
        time_embedding = get_time_embeddings(timestep).to(device)

        model_input = latents

        if do_cfg:
            # (Batch_size, 4, Latents_Height, Latents_Width) -> (2 * Batch_size, 4, Latents_Height, Latents_Width, 1)
            model_input = model_input.repeat(2,1,1,1)
            
        model_output = diffusion(model_input, context, time_embedding)

        if do_cfg:
            output_cond, output_uncond = model_output.chunk(2)
            # combine them using the formula
            model_output = cfg_scale * (output_cond - output_uncond) + output_uncond


        # Remove noise predicted by the unet
        latents = sampler.step(timestep, latents, model_output)

    to_idle(diffusion)

    decoder = models["decoder"]

    decoder.to(device)
    
    images = decoder(latents)

    to_idle(decoder)

    images = rescale(images, (-1,1), (0,255), clamp = True)
    images = images.permute(0,2,3,1).cpu().numpy().astype(np.uint8)
    return images[0]

def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min

    if clamp:
        x = x.clamp(new_min, new_max)

    return x

# this is the positional encoding based on the transformer models
def get_time_embeddings(timestep):
    # define frequency of sine and cosine
    freqs = torch.pow(10000, -torch.arange(start = 0, end = 160, dtype= torch.float32)/160)

    # create a tensor of shape (1, 160)
    x = torch.tensor([timestep], dtype = torch.float32)[:, None] * freqs[None]

    return torch.cat([torch.cos(x), torch.sin(x)], dim = -1)