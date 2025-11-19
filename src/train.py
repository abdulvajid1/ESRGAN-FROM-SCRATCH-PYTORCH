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

logging.basicConfig(
    # filename="training.log",
    level=logging.INFO,
    datefmt="[%X]",                # optional time format
    handlers=[RichHandler()]
    )


def train(generator, discriminator, genr_optimizer: torch.optim.Adam, disc_optimizer: torch.optim.Adam, dataloader, device):
    # setup progres bar
    progress_bar = tqdm.tqdm(dataloader, dynamic_ncols=True)

    for step, (high, low) in enumerate(progress_bar):
        low = low.to(device)
        high = high.to(device)

        with torch.autocast(device_type=device, dtype=torch.bfloat16):

            fake = generator(low)
            fake_critic = discriminator(fake.detach())
            real_critic = discriminator(high)

            fake_mean = fake_critic.mean()
            real_mean = real_critic.mean()


            disc_relative_loss = (
                - F.logsigmoid(real_critic - fake_mean).mean()
                - F.logsigmoid(-(fake_critic - real_mean)).mean()
            )

            disc_optimizer.zero_grad()
            disc_relative_loss.backward()
            disc_optimizer.step()

        with torch.autocast(device_type=device, dtype=torch.bfloat16):

            real_critic = discriminator(high)
            fake_critic = discriminator(fake)

            fake_mean = fake_critic.mean()
            real_mean = real_critic.mean()

            gen_relative_loss = (
                - F.logsigmoid(fake_critic - real_mean).mean()
                - F.logsigmoid(-(real_critic - fake_mean)).mean()
            )

            vg_loss = vgg_loss(high, fake)
            l1_loss = F.l1_loss(high, fake)

            genr_loss = (
                0.5 * gen_relative_loss +
                0.2 * vg_loss +
                0.9 * l1_loss
            )

            genr_optimizer.zero_grad()
            genr_loss.backward()
            genr_optimizer.step()


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def main(args):

    device = get_device()
    logging.info(f"Available Device selected to {device}")

    dataloader = get_dataloader(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_memory)
    logging.info("got dataloader")

    generator = Generator(input_dim=3, num_res_blocks=5)
    discriminator = Discriminator(input_dim=3, num_res_block=3)
    generator = torch.compile(generator)
    discriminator = torch.compile(discriminator)
    logging.info("Loaded both Models")

    genr_optimizer = optim.Adam(generator.parameters(), lr=config.genr_LR)
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=config.disc_LR)
    logging.info("Setup both Optimizer")

    for epoch in range(1, config.NUM_EPOCHS+1):
        train(generator, discriminator, genr_optimizer, disc_optimizer, dataloader, device)



if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    args.add_argument("--num_workers", type=int, default=config.DATALOADER_WORKERS)
    args.add_argument("--pin_memory", "-p", action='store_true')
    args = args.parse_args()

    vgg_loss = VGGLoss()
    main(args)
