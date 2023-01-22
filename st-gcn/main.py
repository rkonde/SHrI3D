import os
import sys

import numpy as np
import torch.optim as optim
import torch.utils.data as data

from config import get_args
from metric import accuracy
from model import *

args = get_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

wlasl_dataset = 100
data_directory = '../WLASL_reduced_skeleton/WLASL'

train_label = torch.tensor(np.load(data_directory + str(wlasl_dataset) + '/train_labels_' + str(wlasl_dataset) + '.npy'))

file_paths = []
for root, directories, files in os.walk(data_directory + str(wlasl_dataset) + '/train'):
    for filename in files:
        filepath = os.path.join(root, filename)
        file_paths.append(filepath)

train_tensor = []

for vid_keypoints_path in file_paths:
    vid_keypoints = np.load(vid_keypoints_path)
    train_tensor.append(vid_keypoints)

train_tensor = torch.tensor(train_tensor)

test_label = torch.tensor(np.load(data_directory + str(wlasl_dataset) + '/test_labels_' + str(wlasl_dataset) + '.npy'))

file_paths = []
for root, directories, files in os.walk(data_directory + str(wlasl_dataset) + '/test'):
    for filename in files:
        filepath = os.path.join(root, filename)
        file_paths.append(filepath)

test_tensor = []

for vid_keypoints_path in file_paths:
    vid_keypoints = np.load(vid_keypoints_path)
    test_tensor.append(vid_keypoints)

test_tensor = torch.tensor(test_tensor)

eval_label = torch.tensor(np.load(data_directory + str(wlasl_dataset) + '/eval_labels_' + str(wlasl_dataset) + '.npy'))

file_paths = []
for root, directories, files in os.walk(data_directory + str(wlasl_dataset) + '/val'):
    for filename in files:
        filepath = os.path.join(root, filename)
        file_paths.append(filepath)

eval_tensor = []

for vid_keypoints_path in file_paths:
    vid_keypoints = np.load(vid_keypoints_path)
    eval_tensor.append(vid_keypoints)

eval_tensor = torch.tensor(eval_tensor)

train_loader = data.DataLoader(data.TensorDataset(train_tensor.to(device)), batch_size=args.batch_size, shuffle=False)
test_loader = data.DataLoader(data.TensorDataset(test_tensor.to(device)), batch_size=args.batch_size, shuffle=False)
valid_loader = data.DataLoader(data.TensorDataset(eval_tensor.to(device)), batch_size=args.batch_size, shuffle=False)
train_label = train_label.to(device)
test_label = test_label.to(device)
eval_label = eval_label.to(device)

reduced_skeleton_adjacency_matrix = [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                                     [1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]]

reduced_skeleton_adjacency_matrix = torch.from_numpy(np.asarray(reduced_skeleton_adjacency_matrix)).to(device)

model = GGCN(reduced_skeleton_adjacency_matrix, train_tensor.size(3), args.num_classes, [train_tensor.size(3), train_tensor.size(3) * 3], [train_tensor.size(3) * 3, 256, 512, 1024], args.feat_dims, args.dropout_rate)
# model = torch.load('wlasl_100_reduced_2412.pt')
if device == 'cuda':
    model.cuda()

num_params = 0
for p in model.parameters():
    num_params += p.numel()
# print(model)
# print('The number of parameters: {}'.format(num_params))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=[args.beta1, args.beta2], weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

best_epoch = 0
best_acc = 0


def train():
    global best_epoch, best_acc

    if args.start_epoch:
        model.load_state_dict(torch.load(os.path.join(args.model_path, 'model-%d.pkl' % (args.start_epoch))))

    # Training
    for epoch in range(args.start_epoch, args.num_epochs):
        train_loss = 0
        train_acc = 0
        # scheduler.step()
        model.train()
        for i, x in enumerate(train_loader):
            logit = model(x[0].float())
            target = train_label[i]
            target = target.to(torch.int64)

            loss = criterion(logit, target.view(1))

            model.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += accuracy(logit, target.view(1))

        print('[epoch', epoch + 1, '] Train loss:', train_loss / i, 'Train Acc:', train_acc / i)

        if (epoch + 1) % args.val_step == 0:
            model.eval()
            val_loss = 0
            val_acc = 0
            train_2_loss = 0
            train_2_acc = 0
            with torch.no_grad():
                for i, x in enumerate(valid_loader):
                    logit = model(x[0].float())
                    target = eval_label[i]
                    target = target.to(torch.int64)

                    val_loss += criterion(logit, target.view(1)).item()
                    val_acc += accuracy(logit, target.view(1))

            print('Val loss:', val_loss / i, 'Val Acc:', val_acc / i)

            with torch.no_grad():
                for i, x in enumerate(test_loader):
                    logit = model(x[0].float())
                    target = test_label[i]
                    target = target.to(torch.int64)

                    train_2_loss += criterion(logit, target.view(1)).item()
                    train_2_acc += accuracy(logit, target.view(1))

                if best_acc <= (train_2_acc / i):
                    best_acc = (train_2_acc / i)
                    torch.save(model, 'slr.pt')

            print('Test loss:', train_2_loss / i, 'Test Acc:', train_2_acc / i)


def test():
    global best_epoch

    # model.load_state_dict(torch.load(os.path.join(args.model_path, 'model-%d.pkl' % best_epoch)))
    model = torch.load('wlasl_100_reduced_2412.pt')
    # print("load model from 'model-%d.pkl'" % best_epoch)

    model.eval()
    test_loss = 0
    test_acc = 0
    arr = []
    with torch.no_grad():
        for i, x in enumerate(test_loader):
            logit = model(x[0].float())
            arr.append(logit)
            target = test_label[i]
            target = target.to(torch.int64)

            test_loss += criterion(logit, target.view(1)).item()
            test_acc += accuracy(logit, target.view(1))
    torch.save(arr, 'skeleton_100.pt')
    print('Test loss:', test_loss / i, 'Test Acc:', test_acc / i)


if __name__ == '__main__':
    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        best_epoch = args.test_epoch
    test()
