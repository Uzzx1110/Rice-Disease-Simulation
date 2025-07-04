"""
Training for CycleGAN

Programmed by Aladdin Persson <aladdin.persson at hotmail dot com>
* 2020-11-05: Initial coding
* 2022-12-21: Small revision of code, checked that it works with latest PyTorch version
"""

import torch
from dataset import HealthyDiseasedDataset
import sys
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_model import Discriminator
from generator_model import Generator


def train_fn(
    disc_H, disc_D, gen_D, gen_H, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler
):
    H_reals = 0
    H_fakes = 0
    loop = tqdm(loader, leave=True)

    for idx, (diseased, healthy) in enumerate(loop):
        diseased = diseased.to(config.DEVICE)
        healthy = healthy.to(config.DEVICE)

        # Train Discriminators H and Z
        with torch.cuda.amp.autocast():
            fake_healthy = gen_H(diseased)
            D_H_real = disc_H(healthy)
            D_H_fake = disc_H(fake_healthy.detach())
            H_reals += D_H_real.mean().item()
            H_fakes += D_H_fake.mean().item()
            D_H_real_loss = mse(D_H_real, torch.ones_like(D_H_real))
            D_H_fake_loss = mse(D_H_fake, torch.zeros_like(D_H_fake))
            D_H_loss = D_H_real_loss + D_H_fake_loss

            fake_diseased = gen_D(healthy)
            D_Z_real = disc_D(diseased)
            D_Z_fake = disc_D(fake_diseased.detach())
            D_Z_real_loss = mse(D_Z_real, torch.ones_like(D_Z_real))
            D_Z_fake_loss = mse(D_Z_fake, torch.zeros_like(D_Z_fake))
            D_Z_loss = D_Z_real_loss + D_Z_fake_loss

            # put it togethor
            D_loss = (D_H_loss + D_Z_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators H and Z
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            D_H_fake = disc_H(fake_healthy)
            D_Z_fake = disc_D(fake_diseased)
            loss_G_H = mse(D_H_fake, torch.ones_like(D_H_fake))
            loss_G_Z = mse(D_Z_fake, torch.ones_like(D_Z_fake))

            # cycle loss
            cycle_diseased = gen_D(fake_healthy)
            cycle_healthy = gen_H(fake_diseased)
            cycle_diseased_loss = l1(diseased, cycle_diseased)
            cycle_healthy_loss = l1(healthy, cycle_healthy)

            # identity loss (remove these for efficiency if you set lambda_identity=0)
            identity_diseased = gen_D(diseased)
            identity_healthy = gen_H(healthy)
            identity_diseased_loss = l1(diseased, identity_diseased)
            identity_healthy_loss = l1(healthy, identity_healthy)

            # add all togethor
            G_loss = (
                loss_G_Z
                + loss_G_H
                + cycle_diseased_loss * config.LAMBDA_CYCLE
                + cycle_healthy_loss * config.LAMBDA_CYCLE
                + identity_healthy_loss * config.LAMBDA_IDENTITY
                + identity_diseased_loss * config.LAMBDA_IDENTITY
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 200 == 0:
            save_image(fake_healthy * 0.5 + 0.5, f"saved_images/healthy_({idx}).jpg")
            save_image(fake_diseased * 0.5 + 0.5, f"saved_images/diseased_({idx}).jpg")

        loop.set_postfix(H_real=H_reals / (idx + 1), H_fake=H_fakes / (idx + 1))


def main():
    disc_H = Discriminator(in_channels=3).to(config.DEVICE)
    disc_D = Discriminator(in_channels=3).to(config.DEVICE)
    gen_D = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_H = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    opt_disc = optim.Adam(
        list(disc_H.parameters()) + list(disc_D.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_D.parameters()) + list(gen_H.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_H,
            gen_H,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_D,
            gen_D,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_H,
            disc_H,
            opt_disc,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_D,
            disc_D,
            opt_disc,
            config.LEARNING_RATE,
        )

    dataset = HealthyDiseasedDataset(
        root_healthy=config.TRAIN_DIR + "/healthy",
        root_diseased=config.TRAIN_DIR + "/bacterial_leaf_blight",
        transform=config.transforms,
    )
    val_dataset = HealthyDiseasedDataset(
        root_healthy=r"E:\Github repos\Rice-Disease-Simulation\DiseaseSimulation-CycleGAN\data\val\healthy",
        root_diseased=r"E:\Github repos\Rice-Disease-Simulation\DiseaseSimulation-CycleGAN\data\val\bacterial_leaf_blight",
        transform=config.transforms,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        train_fn(
            disc_H,
            disc_D,
            gen_D,
            gen_H,
            loader,
            opt_disc,
            opt_gen,
            L1,
            mse,
            d_scaler,
            g_scaler,
        )

        if config.SAVE_MODEL:
            save_checkpoint(gen_H, opt_gen, filename=config.CHECKPOINT_GEN_H)
            save_checkpoint(gen_D, opt_gen, filename=config.CHECKPOINT_GEN_D)
            save_checkpoint(disc_H, opt_disc, filename=config.CHECKPOINT_CRITIC_H)
            save_checkpoint(disc_D, opt_disc, filename=config.CHECKPOINT_CRITIC_D)


if __name__ == "__main__":
    main()
