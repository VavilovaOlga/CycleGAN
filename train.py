import torch
from tqdm import tqdm
import numpy as np
from pathlib import Path
import argparse
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch.nn as nn
import itertools
import pandas as pd

from model import dataset
from model import cycleGAN
from model import utils

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="horse2zebra",
                    help="dataset name. Possible values: \
horse2zebra, photo2portrait. (default: horse2zebra)")

parser.add_argument("--epochs", type=int, default=100,
                    help="number of epochs")

parser.add_argument("--load_weights_path", type=str, default=None,
                    help="path to weights to continuer training")

parser.add_argument("--lr", type=float, default="0.0002",
                    help="lerning rate")

parser.add_argument("--epoch_decay", type=int, default="60",
                    help="epoch to start descrease learning rate")

parser.add_argument("--lambda_identity", type=int, default="5",
                    help="weight for identity loss")

parser.add_argument("--lambda_cycle", type=int, default="10",
                    help="weight for cycle loss")

args = parser.parse_args()


if args.dataset == "horse2zebra":
    trainA_images = list(Path('datasets/horse2zebra/trainA').rglob('*.jpg'))
    trainB_images = list(Path('datasets/horse2zebra/trainB').rglob('*.jpg'))
    testA_images = list(Path('datasets/horse2zebra/testA').rglob('*.jpg'))
    testB_images = list(Path('datasets/horse2zebra/testB').rglob('*.jpg'))

    trainA_dataset = dataset.ImageDataset(trainA_images)
    testA_dataset = dataset.ImageDataset(testA_images)
    trainB_dataset = dataset.ImageDataset(trainB_images)
    testB_dataset = dataset.ImageDataset(testB_images)


if args.dataset == "photo2portrait":
    A_DIR = Path('datasets/faces_dataset_small')
    B_DIR = Path('datasets/original_sketch')
    A_files = list(A_DIR.rglob('*.png'))
    B_files = list(B_DIR.rglob('*.jpg'))

    trainA_images, testA_images = train_test_split(A_files, test_size=0.1)
    trainB_images, testB_images = train_test_split(B_files, test_size=0.1)

    trainA_dataset = dataset.ImageDataset(trainA_images)
    testA_dataset = dataset.ImageDataset(testA_images)
    trainB_dataset = dataset.ImageDataset(trainB_images, resize_size=400,
                                          padding=(0, 50, 0, 0))
    testB_dataset = dataset.ImageDataset(testB_images, resize_size=400,
                                         padding=(0, 50, 0, 0))

trainA_loader = DataLoader(trainA_dataset, batch_size=1, shuffle=True)
testA_loader = DataLoader(testA_dataset, batch_size=4, shuffle=True)
trainB_loader = DataLoader(trainB_dataset, batch_size=1, shuffle=True)
testB_loader = DataLoader(testB_dataset, batch_size=4, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


model = {
    'AB_generator': cycleGAN.generator().to(device),
    'BA_generator': cycleGAN.generator().to(device),
    'A_discriminator': cycleGAN.discriminator().to(device),
    'B_discriminator': cycleGAN.discriminator().to(device)
}

criterion = {
    "generator": nn.MSELoss(),
    "discriminator": nn.MSELoss(),
    "cycle_loss": nn.L1Loss(),
    "identity_loss": nn.L1Loss()
}

optimizer = {
    "generator": torch.optim.Adam(itertools.chain(
        model["AB_generator"].parameters(),
        model["BA_generator"].parameters()), lr=args.lr, betas=(0.5, 0.999)),
    "A_discriminator": torch.optim.Adam(
        model["A_discriminator"].parameters(), lr=args.lr, betas=(0.5, 0.999)),
    "B_discriminator": torch.optim.Adam(
        model["B_discriminator"].parameters(), lr=args.lr, betas=(0.5, 0.999))
}

scheduler = {
     "generator": torch.optim.lr_scheduler.StepLR(optimizer["generator"],
                                                  step_size=10, gamma=0.5),
     "A_discriminator": torch.optim.lr_scheduler.StepLR(
         optimizer["A_discriminator"], step_size=10, gamma=0.5),
     "B_discriminator": torch.optim.lr_scheduler.StepLR(
         optimizer["B_discriminator"], step_size=10, gamma=0.5)
 }


if args.load_weights_path:
    weights_path = Path(args.load_weights_path)
    model["AB_generator"].load_state_dict(torch.load(
        weights_path/'AB_generator.pth', map_location=device))
    model["BA_generator"].load_state_dict(torch.load(
        weights_path/'BA_generator.pth', map_location=device))
    model["A_discriminator"].load_state_dict(torch.load(
        weights_path/'A_discriminator.pth', map_location=device))
    model["B_discriminator"].load_state_dict(torch.load(
        weights_path/'B_discriminator.pth', map_location=device))


model["A_discriminator"].train()
model["B_discriminator"].train()

# Losses
AB_generator_losses = []
BA_generator_losses = []
A_discriminator_losses = []
B_discriminator_losses = []
AB_cycle_losses = []
BA_cycle_losses = []
AB_identity_losses = []
BA_identity_losses = []

fake_A_buffer = utils.Buffer()
fake_B_buffer = utils.Buffer()

test_A = next(iter(testA_loader)).to(device)


for epoch in range(args.epochs):

    model["AB_generator"].train()
    model["BA_generator"].train()

    AB_generator_losses_epoch = []
    BA_generator_losses_epoch = []
    A_discriminator_losses_epoch = []
    B_discriminator_losses_epoch = []
    AB_cycle_losses_epoch = []
    BA_cycle_losses_epoch = []
    AB_identity_loss_epoch = []
    BA_identity_loss_epoch = []

    # training
    for real_A, real_B in tqdm(zip(trainA_loader, trainB_loader),
                               total=min(len(trainA_loader),
                                         len(trainB_loader))):

        real_A = real_A.to(device)
        real_B = real_B.to(device)

        # train generator G
        optimizer["generator"].zero_grad()

        # Identity loss
        identity_B = model["AB_generator"](real_B)
        AB_identity_loss = criterion["identity_loss"](identity_B, real_B)

        identity_A = model["BA_generator"](real_A)
        BA_identity_loss = criterion["identity_loss"](identity_A, real_A)

        # AB
        fake_B = model["AB_generator"](real_A)
        fake_B_preds = model["B_discriminator"](fake_B)
        targets = torch.ones(len(fake_B_preds), 1, device=device)
        AB_generator_loss = criterion["generator"](fake_B_preds, targets)

        fake_A_from_B = model["BA_generator"](fake_B)
        AB_cycle_loss = criterion["cycle_loss"](fake_A_from_B, real_A)

        # BA
        fake_A = model["BA_generator"](real_B)
        fake_A_preds = model["A_discriminator"](fake_A)
        targets = torch.ones(len(fake_A_preds), 1, device=device)
        BA_generator_loss = criterion["generator"](fake_A_preds, targets)

        fake_B_from_A = model["AB_generator"](fake_A)
        BA_cycle_loss = criterion["cycle_loss"](fake_B_from_A, real_B)

        # Update weights
        generator_loss = AB_generator_loss + BA_generator_loss
        + args.lambda_cycle*(AB_cycle_loss + BA_cycle_loss)
        + args.lambda_identity*(AB_identity_loss + BA_identity_loss)
        generator_loss.backward()
        optimizer["generator"].step()

        # train discriminator A
        optimizer["A_discriminator"].zero_grad()

        discriminator_A_real_preds = model["A_discriminator"](real_A)
        targets = torch.ones(len(discriminator_A_real_preds), 1, device=device)
        discriminator_A_real_loss = criterion["discriminator"](
            discriminator_A_real_preds, targets)

        fake_A = fake_A_buffer.take_from_buffer(fake_A)
        fake_A_preds = model["A_discriminator"](fake_A)
        targets = torch.zeros(len(fake_A_preds), 1, device=device)
        discriminator_A_fake_loss = criterion["discriminator"](fake_A_preds,
                                                               targets)
        # Update weights
        A_discriminator_loss = (discriminator_A_fake_loss
                                + discriminator_A_real_loss)/2
        A_discriminator_loss.backward()
        optimizer["A_discriminator"].step()

        # train discriminator B
        optimizer["B_discriminator"].zero_grad()

        discriminator_B_real_preds = model["B_discriminator"](real_B)
        targets = torch.ones(len(discriminator_B_real_preds), 1, device=device)
        discriminator_B_real_loss = criterion["discriminator"](
            discriminator_B_real_preds, targets)

        fake_B = fake_B_buffer.take_from_buffer(fake_B)
        fake_B_preds = model["B_discriminator"](fake_B)
        targets = torch.zeros(len(fake_B_preds), 1, device=device)
        discriminator_B_fake_loss = criterion["discriminator"](fake_B_preds,
                                                               targets)

        # Update weights
        B_discriminator_loss = (discriminator_B_fake_loss
                                + discriminator_B_real_loss)/2
        B_discriminator_loss.backward()
        optimizer["B_discriminator"].step()

        # losses per epoch
        AB_generator_losses_epoch.append(AB_generator_loss.item())
        BA_generator_losses_epoch.append(BA_generator_loss.item())
        A_discriminator_losses_epoch.append(A_discriminator_loss.item())
        B_discriminator_losses_epoch.append(B_discriminator_loss.item())
        AB_cycle_losses_epoch.append(AB_cycle_loss.item())
        BA_cycle_losses_epoch.append(BA_cycle_loss.item())
        AB_identity_loss_epoch.append(AB_identity_loss.item())
        BA_identity_loss_epoch.append(BA_identity_loss.item())

    # Record losses
    AB_generator_losses.append(np.mean(AB_generator_losses_epoch))
    BA_generator_losses.append(np.mean(BA_generator_losses_epoch))
    A_discriminator_losses.append(np.mean(A_discriminator_losses_epoch))
    B_discriminator_losses.append(np.mean(B_discriminator_losses_epoch))
    AB_cycle_losses.append(np.mean(AB_cycle_losses_epoch))
    BA_cycle_losses.append(np.mean(BA_cycle_losses_epoch))
    AB_identity_losses.append(np.mean(AB_identity_loss_epoch))
    BA_identity_losses.append(np.mean(BA_identity_loss_epoch))

    if epoch > args.epoch_decay:
        scheduler["generator"].step()
        scheduler["A_discriminator"].step()
        scheduler["B_discriminator"].step()

    # show results

    model["AB_generator"].eval()
    model["BA_generator"].eval()

    with torch.no_grad():
        generated_B = model["AB_generator"](test_A)
        generated_A_from_B = model["BA_generator"](generated_B)

        utils.show_images(test_A.cpu().detach())
        utils.show_images(generated_B.cpu().detach())
        utils.show_images(generated_A_from_B.cpu().detach())

        # Log losses
        print("Epoch [{}/{}], AB_generator: {:.4f}, BA_generator: {:.4f}, \
A_discriminator: {:.4f}, B_discriminator: {:.4f}, AB_cycle_losses: {:.4f}, \
BA_cycle_losses: {:.4f}, AB_identity_losses: {:.4f}, \
BA_identity_losses: {:.4f}".format(
            epoch+1, args.epochs,
            AB_generator_losses[-1], BA_generator_losses[-1],
            A_discriminator_losses[-1], B_discriminator_losses[-1],
            AB_cycle_losses[-1], BA_cycle_losses[-1], AB_identity_losses[-1],
            BA_identity_losses[-1]))

# save weights
torch.save(model["AB_generator"].state_dict(), 'weights/AB_generator.pth')
torch.save(model["BA_generator"].state_dict(), 'weights/BA_generator.pth')
torch.save(model["A_discriminator"].state_dict(),
           'weights/A_discriminator.pth')
torch.save(model["B_discriminator"].state_dict(),
           'weights/B_discriminator.pth')

losses = pd.DataFrame([])
losses['AB_generator_losses'] = AB_generator_losses
losses['BA_generator_losses'] = BA_generator_losses
losses['A_discriminator_losses'] = A_discriminator_losses
losses['B_discriminator_losses'] = B_discriminator_losses
losses['AB_cycle_losses'] = AB_cycle_losses
losses['BA_cycle_losses'] = BA_cycle_losses
losses['AB_identity_losses'] = AB_identity_losses
losses['BA_identity_losses'] = BA_identity_losses
losses.to_csv('losses/losses.csv', index=False)
