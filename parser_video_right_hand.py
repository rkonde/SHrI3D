import os

import cv2
import torch
import torch.nn as nn
from PIL import Image as im
from torchvision import transforms
from torchvision.models import resnet34

from manolayer import ManoLayer

device = torch.device("cuda:0")

test = resnet34()
test.fc = nn.Linear(512, 58)
test.train()


class AE(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = test

        self.decoder = ManoLayer(mano_root='mano/models', flat_hand_mean=False, use_pca=False, ncomps=45,
                                 side='right').to(device)

    def forward(self, x):
        encoded = self.encoder(x)
        encoded_split = torch.split(encoded, [48, 10], dim=-1)
        decoded = self.decoder(encoded_split[0], encoded_split[1])
        return decoded[1]


def read_frames_from_video(path):
    frames = []
    vidcap = cv2.VideoCapture(path)
    success, image = vidcap.read()
    while success:
        frames.append(image)
        success, image = vidcap.read()
    return frames


def get_filepaths(directory):
    file_paths = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)
    return file_paths


def get_outputs(file, output, target):
    return str(file) + "\n" + str(output[0]) + "\n" + str(target) + "\n"


preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

model = torch.load("./wlasl_decoder_model.pt")

images_paths = get_filepaths('right_hand_videos')

losses = []

index = 0
for image_path in images_paths:
    print('Index: ' + str(index))
    index = index + 1

    frames = read_frames_from_video(image_path)

    for i in range(len(frames)):
        file_outputs = open("./right_hand_keypoints/" + image_path.split('\\')[1].split('.')[0] + ".txt", "a")
        input_image = frames[i]
        input_image = im.fromarray(input_image)
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0).to(device)
        reconstructed = model(input_batch)
        file_outputs.write(str(reconstructed) + "\n")
        file_outputs.close()
