from argparse import Namespace

import torch
import torch.nn as nn
from torchvision import transforms
from pycocotools.coco import COCO
from data_loader import get_loader
from model import EncoderCNN, DecoderRNN
import math
import sys

import torch.utils.data as data
import numpy as np
import os
import requests
import time

args = Namespace(
    ## TODO #1: Select appropriate values for the Python variables below.

    # ===================== main params =====================

    # minimum word count threshold (Vinyals et al, 4.3.1)
    vocab_threshold=5,

    # dimensionality of image and word embeddings
    # (Karpathy et al, 2015, 3.1.2)
    embed_size=256,

    # number of features in hidden state of the RNN decoder
    # (Vinyals et al, 4.3.1)
    hidden_size=512,

    # ===================== misc params =====================

    batch_size=128,  # batch size
    vocab_from_file=True,  # if True, load existing vocab file

    # ===================== already set =====================
    num_epochs=3,  # number of training epochs
    save_every=1,  # determines frequency of saving model weights
    print_every=100,  # determines window for printing average loss
    log_file='training_log.txt',  # name of file with saved training loss and perplexity

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)


def get_transforms():
    # (Optional) TODO #2: Amend the image transform below.
    transform_train = transforms.Compose([
        transforms.Resize(256),  # smaller edge of image resized to 256
        transforms.RandomCrop(224),  # get 224x224 crop from random location
        transforms.RandomHorizontalFlip(),  # horizontally flip image with probability=0.5
        transforms.ToTensor(),  # convert the PIL Image to a tensor
        transforms.Normalize((0.485, 0.456, 0.406),  # normalize image for pre-trained model
                             (0.229, 0.224, 0.225))])
    return transform_train


def get_data_loader():
    # Build data loader.
    data_loader = get_loader(transform=get_transforms(),
                             mode='train',
                             batch_size=args.batch_size,
                             vocab_threshold=args.vocab_threshold,
                             vocab_from_file=args.vocab_from_file)
    vocab_size = len(data_loader.dataset.vocab)
    return data_loader, vocab_size


def train_model(data_loader, encoder, decoder, criterion, optimizer):

    # Open the training log file.
    f = open(args.log_file, 'w')

    old_time = time.time()
    # response = requests.request("GET",
    #                             "http://metadata.google.internal/computeMetadata/v1/instance/attributes/keep_alive_token",
    #                             headers={"Metadata-Flavor":"Google"})

    for epoch in range(1, args.num_epochs + 1):

        for i_step in range(1, total_step + 1):

            if time.time() - old_time > 60:
                old_time = time.time()
            #             requests.request("POST",
            #                              "https://nebula.udacity.com/api/v1/remote/keep-alive",
            #                              headers={'Authorization': "STAR " + response.text})

            # Randomly sample a caption length, and sample indices with that length.
            indices = data_loader.dataset.get_train_indices()
            # Create and assign a batch sampler to retrieve a batch with the sampled indices.
            new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
            data_loader.batch_sampler.sampler = new_sampler

            # Obtain the batch.
            images, captions = next(iter(data_loader))

            # Move batch of images and captions to GPU if CUDA is available.
            images = images.to(args.device)
            captions = captions.to(args.device)

            # Zero the gradients.
            decoder.zero_grad()
            encoder.zero_grad()

            # Pass the inputs through the CNN-RNN model.
            features = encoder(images)
            outputs = decoder(features, captions)

            # Calculate the batch loss.
            loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))

            # Backward pass.
            loss.backward()

            # Update the parameters in the optimizer.
            optimizer.step()

            # Get training statistics.
            stats = 'Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f' % (
            epoch, args.num_epochs, i_step, total_step, loss.item(), np.exp(loss.item()))

            # Print training statistics (on same line).
            print('\r' + stats, end="")
            sys.stdout.flush()

            # Print training statistics to file.
            f.write(stats + '\n')
            f.flush()

            # Print training statistics (on different line).
            if i_step % args.print_every == 0:
                print('\r' + stats)

        # Save the weights.
        if epoch % args.save_every == 0:
            torch.save(decoder.state_dict(), os.path.join('./models', 'decoder-%d.pkl' % epoch))
            torch.save(encoder.state_dict(), os.path.join('./models', 'encoder-%d.pkl' % epoch))

    # Close the training log file.
    f.close()


if __name__ == '__main__':
    data_loader, vocab_size = get_data_loader()

    encoder = EncoderCNN(args.embed_size).to(args.device)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, vocab_size).to(args.device)

    criterion = nn.CrossEntropyLoss().to(args.device)
    params = list(decoder.parameters()) + list(encoder.embed.parameters())
    optimizer = torch.optim.Adam(params)
    total_step = math.ceil(len(data_loader.dataset.caption_lengths) / data_loader.batch_sampler.batch_size)

    train_model(data_loader, encoder, decoder, criterion, optimizer)
