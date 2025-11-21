import torch.nn as nn
import torch
from torchvision.models import vgg19
from pathlib import Path
import torchvision

class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg19 = vgg19(pretrained=True).features[:34] # before relu
        for params in self.vgg19.parameters():
            params.required_grad = False
        
        self.mse = nn.MSELoss()
    
    def forward(self, real: torch.Tensor, fake: torch.Tensor):
        real = self.vgg19(real)
        fake = self.vgg19(fake)
        return self.mse(real, fake)
    
def visualize_sample(generator, dataloader, global_step):
    path = Path("GenImages")
    path.mkdir(parents=True, exist_ok=True)

    # get one batch (high_res, low_res)
    high_batch, low_batch = next(iter(dataloader))
    high, low = high_batch[0].unsqueeze(0), low_batch[0].unsqueeze(0)

    # determine device of the generator
    try:
        device = next(generator.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    generator.eval()
    with torch.no_grad():
        low = low.to(device)
        gen_high = generator(low)

    # move tensors to cpu for saving
    high = high.detach().cpu()
    gen_high = gen_high.detach().cpu()

    # concatenate originals and generated images so that the first row is the
    # real images and the second row is the generated images
    concat = torch.cat([high, gen_high], dim=0)

    filename = path / f"vis_{global_step}.png"
    torchvision.utils.save_image(concat, str(filename), nrow=high.size(0), normalize=True, scale_each=True)

    generator.train()
    return filename
    

def save_checkpoint(generator: torch.nn.Module, discriminator, gen_optimizer, disc_optimizer, global_step, pretrain=True):
    path = Path("checkpoints")
    path.mkdir(exist_ok=True)

    if pretrain:
        checkpoint = {
            "generator": generator.state_dict(),
            "gen_optimizer": gen_optimizer.state_dict(),
            "global_step": int(global_step),
        }
    else:
        checkpoint = {
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "gen_optimizer": gen_optimizer.state_dict(),
            "disc_optimizer": disc_optimizer.state_dict(),
            "global_step": int(global_step),
        }

    filepath = path / f"checkpoint_{global_step}.ckpt"
    torch.save(checkpoint, filepath)
    return filepath

def load_checkpoint(generator, discriminator, gen_optimizer, disc_optimizer, pretrain=True):
    path = Path("checkpoints")
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {path}")

    all_checkpoint = list(path.glob("*.ckpt"))
    if not all_checkpoint:
        raise FileNotFoundError(f"No checkpoint files found in: {path}")

    # pick the most recently modified checkpoint
    latest = sorted(all_checkpoint, key=lambda p: p.stat().st_mtime)[-1]
    ckpt = torch.load(latest, map_location=lambda storage, loc: storage)

    # load generator and optimizer states
    if "generator" in ckpt and generator is not None:
        generator.load_state_dict(ckpt["generator"])

    if pretrain:
        if "gen_optimizer" in ckpt and gen_optimizer is not None:
            gen_optimizer.load_state_dict(ckpt["gen_optimizer"])
    else:
        if "discriminator" in ckpt and discriminator is not None:
            discriminator.load_state_dict(ckpt["discriminator"])
        if "gen_optimizer" in ckpt and gen_optimizer is not None:
            gen_optimizer.load_state_dict(ckpt["gen_optimizer"])
        if "disc_optimizer" in ckpt and disc_optimizer is not None:
            disc_optimizer.load_state_dict(ckpt["disc_optimizer"])

    return ckpt.get("global_step", None)
        