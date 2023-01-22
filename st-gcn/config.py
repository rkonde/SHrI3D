import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--train_path', type=str, default='./dataset/train.pkl')
    parser.add_argument('--valid_path', type=str, default='./dataset/valid.pkl')
    parser.add_argument('--test_path', type=str, default='./dataset/test.pkl')
    parser.add_argument('--train_label_path', type=str, default='./dataset/train_label.pkl')
    parser.add_argument('--valid_label_path', type=str, default='./dataset/valid_label.pkl')
    parser.add_argument('--test_label_path', type=str, default='./dataset/test_label.pkl')
    parser.add_argument('--model_path', type=str, default='./models')

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--learning_rate', type=int, default=0.000005)
    parser.add_argument('--beta1', type=int, default=0.5)
    parser.add_argument('--beta2', type=int, default=0.99)
    parser.add_argument('--dropout_rate', type=int, default=0.25)
    parser.add_argument('--weight_decay', type=int, default=0.00001)

    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--test_epoch', type=int, default=0)
    parser.add_argument('--val_step', type=int, default=1)

    parser.add_argument('--num_classes', type=int, default=100)
    parser.add_argument('--feat_dims', type=int, default=300)

    args = parser.parse_args()

    return args

# whole 3d body graph, max accuracy 22,17%
# reduced 3d body graph, max accuracy %
# 2d body graph, max accuracy 21,7%