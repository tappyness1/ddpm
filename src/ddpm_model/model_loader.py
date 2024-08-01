from src.ddpm_model.clip import CLIP
from src.ddpm_model.vae import EncoderVAE, DecoderVAE
from src.ddpm_model.diffusion import Diffusion
import src.ddpm_model.model_converter as model_converter


def preload_models_from_standard_weights(ckpt_path, device):
    state_dict = model_converter.load_from_standard_weights(ckpt_path, device)

    encoder = EncoderVAE().to(device)
    encoder.load_state_dict(state_dict['encoder'], strict=True)

    decoder = DecoderVAE().to(device)
    decoder.load_state_dict(state_dict['decoder'], strict=True)

    diffusion = Diffusion().to(device)
    diffusion.load_state_dict(state_dict['diffusion'], strict=True)

    clip = CLIP().to(device)
    clip.load_state_dict(state_dict['clip'], strict=True)

    return {
        'clip': clip,
        'encoder': encoder,
        'decoder': decoder,
        'diffusion': diffusion,
    }

if __name__ == "__main__":
    weights = preload_models_from_standard_weights('./model_weights/v1-5-pruned-emaonly.ckpt', 'cpu')

