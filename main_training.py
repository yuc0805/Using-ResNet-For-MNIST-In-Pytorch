import argparse
import numpy as np
import os
import datetime
import time
from pathlib import Path

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim

from datasets import build_dataset
from engine_main import train_one_epoch, evaluate
import model_resnet

def get_args_parser():
    parser = argparse.ArgumentParser('Resnet training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size')
    parser.add_argument('--epochs', default=20, type=int)
    
    #Model parameters
    parser.add_argument('--input_size', default=28, type=int,
                        help='images input size')
    parser.add_argument('--num_classes', default=10, type=int,
                        help='number of classes')
    parser.add_argument('--model', default='resnet_18', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--in_channels', default=1, type=int,
                        help='size of channels')


    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.2, metavar='PCT',
                        help='Color jitter factor')
    parser.add_argument('--random_affine', nargs='*', default=[20, (0.1, 0.1), (0.9, 1.1)],
                    help='random affine arguments')
    
    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (absolute lr)')

    # Dataset parameters
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='mps',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)

    # Training parameters
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')

    return parser


def main(args):
    device = torch.device(args.device)

    #fix seed
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # load data
    dataset_train = build_dataset(is_train=True, args=args)
    dataset_val = build_dataset(is_train=False, args=args)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,batch_size=args.batch_size, shuffle=True)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,batch_size=args.batch_size, shuffle=True)

    # define the model
    model = model_resnet.__dict__[args.model](
        in_channels = args.in_channels,
        num_classes = args.num_classes
    )

    model.to(device)
    print("Model = %s" % str(model))
    
    # training
    batch_size = args.batch_size
    print("lr: %.2e" % args.lr)

    #optimizer
    optimizer = torch.optim.Adam(model.parameters(),args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    if args.eval:
        evaluate(data_loader_val,model,device)
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    
    for epoch in range(args.start_epoch,args.epochs):
        train_one_epoch(model,criterion,data_loader_train,
                        optimizer,device,epoch)
        evaluate(data_loader_val,model,device)
       
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # save model
    if args.output_dir:
        model_path = os.path.join(args.output_dir, 'resnet_model.pth')
        torch.save(model.state_dict(),model_path)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True,exist_ok=True)
    main(args)