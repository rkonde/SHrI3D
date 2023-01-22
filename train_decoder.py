import json
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet34

from manolayer import ManoLayer

device = torch.device("cuda:0")

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

resnet = resnet34()
resnet.fc = nn.Linear(512, 58)
resnet.train()

ncomps = 45


class AE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = resnet
        self.decoder = ManoLayer(mano_root='mano/models', flat_hand_mean=False, use_pca=False, ncomps=ncomps)

    def forward(self, x):
        encoded = self.encoder(x)
        encoded_split = torch.split(encoded, [48, 10], dim=-1)
        decoded = self.decoder(encoded_split[0], encoded_split[1])
        return decoded[1]


model = AE().to(device)
# model = torch.load("./wlasl_decoder_model.pt")
loss_function = torch.nn.MSELoss().to(device)
# lr=0.00001 then lr=0.000001 then lr=0.0000001
optimizer = torch.optim.Adam(model.encoder.parameters(), lr=0.0000001, weight_decay=0)


def get_filepaths(directory):
    file_paths = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)

    return file_paths


images_paths = get_filepaths('./FreiHAND/freihand_training_images')

f = open('./FreiHAND/training_mano.json')
hand_coordinates = json.load(f)

hand_coordinates_cut = []

for coord in hand_coordinates:
    hand_coordinates_cut.append(coord[0][:58])

epochs = 5
outputs = []
losses = []


def spatial_loss(output, target):
    return torch.norm(torch.sub(output, target))


def get_outputs(file, output, target):
    return str(file) + "\n" + str(output[0]) + "\n" + str(target) + "\n"


previous_avg_loss = 0

print("Samples: " + str(len(images_paths)))

for epoch in range(epochs):
    print('--------Epoch--------')
    print(epoch)

    avg_loss = 0

    for i in range(len(images_paths)):
        if i > 0 and i % 10 == 0:
            optimizer.zero_grad()
        input_image = Image.open(images_paths[i])
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0).to(device)

        # Output of Autoencoder
        reconstructed = model(input_batch)

        actual_arr = torch.from_numpy(np.array([hand_coordinates_cut[i % 32560]])).type(torch.float32).to(device)
        actual_arr_split = torch.split(actual_arr, [48, 10], dim=-1)
        actual_hand_joints = model.decoder(actual_arr_split[0], actual_arr_split[1])[1]

        loss = spatial_loss(reconstructed, actual_hand_joints)
        avg_loss += loss.item()

        loss.backward()
        optimizer.step()

    print("Average loss for this epoch:")
    avg_loss = avg_loss / len(images_paths)
    print("Current avg loss " + str(avg_loss))
    previous_avg_loss = avg_loss

torch.save(model, 'wlasl_decoder_model_trained.pt')

f.close()
