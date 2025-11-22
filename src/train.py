import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from argparse import ArgumentParser
import logging
from rich.logging import RichHandler
import tqdm

import config
from dataset import get_dataloader
from model import Generator, Discriminator
from utils import VGGLoss

from utils import load_checkpoint, save_checkpoint, visualize_sample

logging.basicConfig(
    # filename="training.log",
    level=logging.INFO,
    datefmt="[%X]",                # optional time format
    handlers=[RichHandler()]
    )

torch.set_float32_matmul_precision('high')

def train(generator, discriminator, genr_optimizer: torch.optim.Adam, disc_optimizer: torch.optim.Adam, dataloader, device, epoch, global_step, save_step):
    # setup progres bar
    progress_bar = tqdm.tqdm(dataloader, dynamic_ncols=True)
    data_len = len(dataloader)

    if global_step == None:
        global_step = (epoch * data_len + step)

    for step, (high, low) in enumerate(progress_bar):
        global_step += 1
        low = low.to(device)
        high = high.to(device)

        # with torch.autocast(device_type=device, dtype=torch.bfloat16):

        #     fake = generator(low)
        #     fake_critic = discriminator(fake.detach())
        #     real_critic = discriminator(high)

        #     fake_mean = fake_critic.mean()
        #     real_mean = real_critic.mean()


        #     disc_relative_loss = (
        #         - F.logsigmoid(real_critic - fake_mean).mean()
        #         - F.logsigmoid(-(fake_critic - real_mean)).mean()
        #     )

        #     disc_optimizer.zero_grad()
        #     disc_relative_loss.backward()
        #     disc_optimizer.step()

        with torch.autocast(device_type=device, dtype=torch.bfloat16):

            fake = generator(low) # this should removed for finetuning
            # real_critic = discriminator(high)
            # fake_critic = discriminator(fake)

            # fake_mean = fake_critic.mean()
            # real_mean = real_critic.mean()

            # gen_relative_loss = (
            #     - F.logsigmoid(fake_critic - real_mean).mean()
            #     - F.logsigmoid(-(real_critic - fake_mean)).mean()
            # )

            # vg_loss = vgg_loss(high, fake)
            l1_loss = F.l1_loss(high, fake)

            # genr_loss = (
            #     0.5 * gen_relative_loss +
            #     0.2 * vg_loss +
            #     0.9 * l1_loss
            # )

            genr_optimizer.zero_grad()
            l1_loss.backward()
            genr_optimizer.step()


        if global_step % save_step == 0:
            visualize_sample(generator=generator, dataloader=dataloader, global_step=global_step)
            save_checkpoint(generator, discriminator, genr_optimizer, disc_optimizer, global_step=global_step, pretrain=True)
            logging.info(f"Saved the img & Model at {global_step}")

        progress_bar.set_postfix({
            "genr_loss": f"{l1_loss.item(): .5f}",
            # "desc_loss": f"{disc_relative_loss.item(): .5f}", # TODO uncomment for finetune
        })
    
    return global_step
        


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def main(args):

    device = get_device()
    logging.info(f"Available Device selected to {device}")

    dataloader = get_dataloader(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_memory)
    logging.info("got dataloader")

    generator = Generator(input_dim=3, num_res_blocks=5).to(device)
    discriminator = Discriminator(input_dim=3, num_res_block=3).to(device)
    generator = torch.compile(generator).train()
    discriminator = torch.compile(discriminator).train()
    logging.info("Loaded both Models")

    genr_optimizer = optim.Adam(generator.parameters(), lr=config.genr_LR, fused=True)
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=config.disc_LR, fused=True)
    logging.info("Setup both Optimizer")
    global_step = None
    if args.load:
        global_step, latest = load_checkpoint(generator, discriminator, genr_optimizer, disc_optimizer, pretrain=True)
        logging.info(f"Restarting from Global step {global_step}")
        logging.info(f"Loading from latest checkpoint {latest}")
    else:
        logging.info(f"Global Step 1")

    for epoch in range(0, config.NUM_EPOCHS):
        global_step = train(generator, discriminator, genr_optimizer, disc_optimizer, dataloader, device, epoch, global_step=global_step, save_step=100)


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    args.add_argument("--num_workers", type=int, default=config.DATALOADER_WORKERS)
    args.add_argument("--pin_memory", "-p", action='store_true')
    args.add_argument("--load", "-l", action='store_true')
    args = args.parse_args()

    # vgg_loss = VGGLoss().to(device=get_device())
    main(args)
