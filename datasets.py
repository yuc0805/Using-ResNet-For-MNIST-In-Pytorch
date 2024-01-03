import os
import PIL
from torchvision import datasets, transforms


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    dataset = datasets.__dict__[args.datasets]('../data', train=is_train, download=True,
                       transform=transform)
    
    print(dataset)
    return dataset



def build_transform(is_train, args):
    mean = 0.1307 #default MNIST Mean
    std = 0.3081  #default MNIST std
    # train transform
    if is_train:
        degrees, translate, scale = args.random_affine
        transform = transforms.Compose([
        transforms.RandomAffine(degrees=degrees, translate=translate, scale=scale),
        transforms.ColorJitter(args.color_jitter),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,std=std)
        ])
    
    #eval transform
    else:
        transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    return transform

