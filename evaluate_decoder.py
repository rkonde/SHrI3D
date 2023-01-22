import json
import os

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet34
import cv2
from PIL import Image as im

from manolayer import ManoLayer

# device = torch.device("cuda:0")
device = torch.device("cuda:0")

joint_translation_mano_to_real = [20, 17, 16, 18, 19, 1, 0, 2, 3, 5, 4, 6, 7, 13, 12, 14, 15, 9, 8, 10, 11]
joint_translation_real_to_mano = [6, 5, 7, 8, 10, 9, 11, 12, 18, 17, 19, 20, 14, 13, 15, 16, 2, 1, 3, 4, 0]

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test = resnet34()
test.fc = nn.Linear(512, 58)
test.train()


class AE(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = test

        manolayertest = ManoLayer(mano_root='mano/models', flat_hand_mean=False, use_pca=False, ncomps=45).to(device)
        self.decoder = manolayertest

    def forward(self, x):
        encoded = self.encoder(x)
        encoded_split = torch.split(encoded, [48, 10], dim=-1)
        decoded = self.decoder(encoded_split[0], encoded_split[1])
        return decoded[1]


# Model Initialization
model = torch.load("./wlasl_decoder_model.pt")

frames = []


def get_filepaths(directory):
    file_paths = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)
    return file_paths


images_paths = get_filepaths('./FreiHANDrgb_eval')

f = open('./FreiHAND/evaluation_mano.json')

hand_coordinates = json.load(f)

hand_coordinates_cut = []

for coord in hand_coordinates:
    hand_coordinates_cut.append(coord[0][:58])

epochs = 1
outputs = []
losses = []


def spatial_loss(output, target):
    return torch.norm(torch.sub(output, target))


print("Samples: " + str(len(images_paths)))

file_outputs = open("outputs_parser.txt", "a")
file_losess = open("losses_parser.txt", "a")


def get_outputs(file, output, target):
    return str(file) + "\n" + str(output[0]) + "\n" + str(target) + "\n"


for epoch in range(epochs):
    print('--------Epoch--------')
    print(epoch)
    avg_loss = 0
    for i in range(len(images_paths)):
        print(i)
        input_image = im.open(images_paths[i])
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0).to(device)

        # Output of Autoencoder
        reconstructed = model(input_batch)
        actual_arr = torch.from_numpy(np.array([hand_coordinates_cut[i]])).type(torch.float32).to(device)
        tesadas = torch.split(actual_arr, [48, 10], dim=-1)

        xd = model.decoder(tesadas[0].to(device), tesadas[1].to(device))
        loss = spatial_loss(reconstructed, xd[1])
        avg_loss += loss.item()

        file_outputs.write(get_outputs(images_paths[i], reconstructed, xd[1]))
        file_losess.write(str(images_paths[i]) + ", " + str(loss) + "\n")

    print("Average loss for this epoch:")
    avg_loss = avg_loss / len(images_paths)
    print(avg_loss)

file_outputs.close()
file_losess.close()
