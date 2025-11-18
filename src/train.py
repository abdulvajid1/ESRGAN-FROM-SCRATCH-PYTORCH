import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import ArgumentParser
import config

from model import Generator, Discriminator

def train(generator, discriminator):
    pass

def main():
    args = ArgumentParser()
    args.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    args.add_argument("--num_workers", type=int, default=config.DATALOADER_WORKERS)

    args.parse_args()