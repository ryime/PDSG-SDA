import numpy as np
import torch
import argparse
import yaml
from utils.attack import set_PDSG, FGSM, PGD, SDA
from utils.encoder import get_encoder
import torchvision
import os
from timm.data import create_loader
from torchvision import transforms
from utils.utils import DatasetSplitter, DatasetWarpper, DVStransform
import logging
from timm.models import create_model
import models.resnet, models.vgg
import errno
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    config_parser = argparse.ArgumentParser(description="Attack Config", add_help=False)

    config_parser.add_argument(
        "-c",
        "--config",
        type=str,
        metavar="FILE",
        help="YAML config file specifying default arguments",
    )

    parser = argparse.ArgumentParser(description='Attacking')

    # testing options
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--batch-size', default=256, type=int)
    parser.add_argument('--T', default=4, type=int, help='simulation steps')
    parser.add_argument('--encoding', default='direct', type=str, help='encoding scheme')
    parser.add_argument('--model', default='spiking_resnet18', help='model name')
    parser.add_argument('--dataset', default='CIFAR10', help='dataset name')
    parser.add_argument('--workers', default=16, type=int, help='number of data loading workers')

    parser.add_argument('--data-path', default='./datasets')
    parser.add_argument('--output-dir', default='./logs/temp')
    parser.add_argument('--resume', type=str, help='model checkpoint')
    
    # attacking options
    parser.add_argument('--attack', default='FGSM', type=str)
    parser.add_argument('--attack_eps', default=8, type=int)

    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
        parser.set_defaults(**cfg)
    args = parser.parse_args(remaining)

    return args

def setup_logger(output_dir):
    logger = logging.getLogger(__name__)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s]%(message)s',
                                  datefmt=r'%Y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(os.path.join(output_dir, 'log.log'))
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)
    return logger

def load_data(
    dataset_dir: str,
    batch_size: int,
    workers: int,
    dataset_type: str,
    T: int,
):
    if dataset_type == 'CIFAR10':
        num_classes = 10
        input_size = (3, 32, 32)
        dataset_test = torchvision.datasets.CIFAR10(root=os.path.join(dataset_dir), train=False,
                                                    download=True)
        data_loader_test = create_loader(
            dataset_test,
            input_size=input_size,
            batch_size=batch_size,
            is_training=False,
            use_prefetcher=True,
            interpolation='bicubic',
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010),
            num_workers=workers,
            crop_pct=1.0,
            pin_memory=True,
        )
        data_loader_attack = data_loader_test
    elif dataset_type == 'CIFAR100':
        num_classes = 100
        input_size = (3, 32, 32)
        dataset_test = torchvision.datasets.CIFAR100(root=os.path.join(dataset_dir), train=False,
                                                     download=True)
        data_loader_test = create_loader(
            dataset_test,
            input_size=input_size,
            batch_size=batch_size,
            is_training=False,
            use_prefetcher=True,
            interpolation='bicubic',
            mean=[n / 255. for n in [129.3, 124.1, 112.4]],
            std=[n / 255. for n in [68.2, 65.4, 70.4]],
            num_workers=workers,
            crop_pct=1.0,
            pin_memory=True,
        )
        data_loader_attack = data_loader_test
    
    elif dataset_type == 'CIFAR10DVS':
        num_classes = 10
        input_size = (2, 128, 128)
        from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
        transform_test = DVStransform(
            transform=transforms.Resize(size=input_size[-2:], antialias=True))
        dataset = CIFAR10DVS(dataset_dir, data_type='frame', frames_number=T, split_by='number')
        dataset_test = DatasetSplitter(dataset, 0.1, False)
        dataset_test = DatasetWarpper(dataset_test, transform_test)
        data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size,
                                                       shuffle=True, num_workers=workers,
                                                       pin_memory=True, drop_last=False)
        data_loader_attack = torch.utils.data.DataLoader(dataset_test, batch_size=1,
                                                       shuffle=True, num_workers=workers,
                                                       pin_memory=True, drop_last=False)
    elif dataset_type == 'DVSGesture':
        num_classes = 11
        input_size = (2, 128, 128)
        from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
        transform_test = DVStransform(
            transform=transforms.Resize(size=input_size[-2:], antialias=True))
        dataset_test = DVS128Gesture(dataset_dir, train=False, data_type='frame', frames_number=T,
                                     split_by='number')
        dataset_test = DatasetWarpper(dataset_test, transform_test)
        data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size,
                                                       shuffle=True, num_workers=workers,
                                                       pin_memory=True, drop_last=False)
        data_loader_attack = torch.utils.data.DataLoader(dataset_test, batch_size=1,
                                                       shuffle=True, num_workers=workers,
                                                       pin_memory=True, drop_last=False)
    elif dataset_type == 'NMNIST':
        num_classes = 10
        input_size = (2, 34, 34)
        from spikingjelly.datasets.n_mnist import NMNIST
        transform_test = DVStransform(
            transform=transforms.Resize(size=input_size[-2:], antialias=True))
        dataset_test = NMNIST(dataset_dir, train=False, data_type='frame', frames_number=T,
                                     split_by='number')
        dataset_test = DatasetWarpper(dataset_test, transform_test)
        data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size,
                                                       shuffle=True, num_workers=workers,
                                                       pin_memory=True, drop_last=False)
        data_loader_attack = torch.utils.data.DataLoader(dataset_test, batch_size=1,
                                                       shuffle=True, num_workers=workers,
                                                       pin_memory=True, drop_last=False)
    elif dataset_type == 'ImageNet':
        num_classes = 1000
        input_size = (3, 224, 224)
        valdir = os.path.join(dataset_dir, 'val')
        dataset_test = torchvision.datasets.ImageFolder(valdir)
        data_loader_test = create_loader(
            dataset_test,
            input_size=input_size,
            batch_size=batch_size,
            is_training=False,
            use_prefetcher=True,
            interpolation='bicubic',
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            num_workers=workers,
            crop_pct=0.95,
            pin_memory=True,
        )
        data_loader_attack = data_loader_test
    else:
        raise ValueError(dataset_type)

    return num_classes, input_size, data_loader_test, data_loader_attack

def main():    
    args = parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    try:
        os.makedirs(args.output_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    logger = setup_logger(args.output_dir)
    logger.info(str(args))
    
    logger.info('------SNN TESTING------')
    dataset_type = args.dataset

    num_classes, input_size, data_loader_test, data_loader_attack = load_data(
        args.data_path, args.batch_size, args.workers, dataset_type, args.T)
     
    net = create_model(
        args.model,
        T=args.T,
        num_classes=num_classes,
        img_size=input_size,
    ).cuda()

    encoder = get_encoder(args.encoding, args.T)
    
    if args.attack == 'FGSM':
        attack_generator = FGSM(net, encoder, args.attack_eps / 255)
    elif args.attack == 'PGD':
        attack_generator = PGD(net, encoder, args.attack_eps / 255)
    elif args.attack == 'SDA':
        attack_generator = SDA(net, batch_limit=args.batch_size)
    else:
        raise NotImplementedError(args.attack)
    
    if args.attack == 'FGSM' or args.attack == 'PGD':
        if dataset_type == 'CIFAR10':
            attack_generator.set_normalization_used(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        elif dataset_type == 'CIFAR100':
            attack_generator.set_normalization_used(mean=[n / 255. for n in [129.3, 124.1, 112.4]], std=[n / 255. for n in [68.2, 65.4, 70.4]])
        elif dataset_type == 'ImageNet':
            attack_generator.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if dataset_type == 'DVSGesture' or dataset_type == 'CIFAR10DVS' or dataset_type == 'NMNIST':
            attack_generator.max_val = None
    
    # adopt PDSG surrogate function        
    net = set_PDSG(net)
    
    state = torch.load(args.resume, map_location=device)
    net.load_state_dict(state['model'])
    
    torch.cuda.empty_cache()
    net.eval()
    
    # attacking static images or dynamic integer frames
    if args.encoding != 'binary':
        correct = 0
        adversarial_correct = 0
        total = 0
        for batch_idx, (images, labels) in enumerate(data_loader_test):
            images, labels = images.to(device), labels.to(device)
            
            # original accuracy
            with torch.no_grad():
                encoded_images = encoder(images.detach())
                outputs = net(encoded_images).mean(0)
                _, predicted = outputs.max(1)
                total += float(labels.size(0))
                correct += float(predicted.eq(labels).sum().item())

            # perform attack
            adversarial_images = attack_generator(images, labels)

            # adversarial accuracy
            with torch.no_grad():
                encoded_images = encoder(adversarial_images)
                outputs = net(encoded_images).mean(0)
                _, predicted = outputs.max(1)
                adversarial_correct += float(predicted.eq(labels).sum().item())
                
            accuracy = 100.0 * correct / total
            adversarial_accuracy = 100.0 * adversarial_correct / total
            ASR = 100.0 * (accuracy - adversarial_accuracy) / accuracy
            
            logger.info(f'Batch:[{batch_idx+1}/{len(data_loader_test)}], Accuracy: {accuracy:.2f}%, Adversarial Accuracy: {adversarial_accuracy:.2f}%, Attack Success Rate: {ASR:.2f}%')
    else:
        total = 0
        correct = 0
        L0_list = []
        
        # original accuracy
        for batch_idx, (images, labels) in enumerate(data_loader_test):
            images, labels = images.to(device), labels.to(device)
            
            with torch.no_grad():
                encoded_images = encoder(images.detach())
                outputs = net(encoded_images).mean(0)
                _, predicted = outputs.max(1)
                
                total += float(labels.size(0))
                correct += float(predicted.eq(labels).sum().item())
        logger.info(f'Accuracy: {100.0 * correct / total}')

        for batch_idx, (images, labels) in enumerate(data_loader_attack):
            images, labels = images.to(device), labels.to(device)
            
            # batch_size=1 in DVS attack
            for b in range(images.shape[0]):
                image, label = images[b].unsqueeze(0), labels[b].unsqueeze(0)
                image = encoder(image)
                with torch.no_grad():
                    output = net(image).mean(0)
                    _, predicted = output.max(1)
                    
                # only attack correctly classified inputs
                if predicted.eq(label).sum().item() == 0:
                    continue
                else:
                    adversarial_image = attack_generator(image, label)
                    L0 = image.not_equal(adversarial_image).sum().cpu().item()
                    L0_list.append(L0)
                    
            L0_array = np.array(L0_list)
            success = L0_array[L0_array > 0].size # if L0=0, attack failed
            L0_200 = 100.0 * L0_array[(L0_array > 0) & (L0_array < 200)].size / (success if success > 0 else 1)
            L0_800 = 100.0 * L0_array[(L0_array > 0) & (L0_array < 800)].size / (success if success > 0 else 1)
            L0_mean = np.mean(L0_array[L0_array > 0])
            L0_median = np.median(L0_array[L0_array > 0])
            logger.info(f'Batch:[{batch_idx+1}/{len(data_loader_attack)}], L0_200: {L0_200:.2f}%, L0_800: {L0_800:.2f}%, L0_mean: {L0_mean:.2f}, L0_median: {L0_median:.2f}')
            if len(L0_array) >= 100: # attack random 100 inputs
                break
                    
    logger.info('------SNN TESTING FINISHED------')
if __name__ == '__main__':
    main()