# Towards Effective and Sparse Adversarial Attack on Spiking Neural Networks via Breaking Invisible Surrogate Gradients (CVPR 2025)

This repo is the official implementation of "Towards Effective and Sparse Adversarial Attack on Spiking Neural Networks via Breaking Invisible Surrogate Gradients" (CVPR 2025)

## Requirements
````
python version: 3.11.9
CUDA version: 12.4
````

````
numpy==2.1.3
PyYAML==6.0.1
spikingjelly==0.0.0.0.14
timm==1.0.11
torch==2.3.0
torchattacks==3.5.1
torchvision==0.18.0
````

## Prepare

````
conda create --name snn_attack python=3.11.9
pip install -r requirements.txt
````

## Download checkpoints
Google Drive: [Download model checkpoints](https://drive.google.com/drive/folders/1c8-D1VkeDGkBm2dEM2uppjsE2Yrp8Nca?usp=sharing)

Hugging Face: [Download model checkpoints](https://huggingface.co/ryime/PDSG-SDA)

## Run

CIFAR10-ResNet18:
````
python test.py -c ./configs/resnet18_cifar10.yaml --data-path your_dataset_path
````

CIFAR10-ResNet18(Adversarially trained):
````
python test.py -c ./configs/resnet18_advtrained_cifar10.yaml --data-path your_dataset_path
````

CIFAR100-ResNet18:
````
python test.py -c ./configs/resnet18_cifar100.yaml --data-path your_dataset_path
````

DVSGesture-VGGSNN:
````
python test.py -c ./configs/vggsnn_dvsgesture_binary.yaml --data-path your_dataset_path
````

CIFAR10DVS-ResNet18:
````
python test.py -c ./configs/resnet18_cifar10dvs_binary.yaml --data-path your_dataset_path
````

## Acknowledgments
The frame of this code is altered from [SpikingResformer](https://github.com/xyshi2000/SpikingResformer). We thank the authors for their contribution.

## Citation
If you find this paper useful, please consider staring this repository and citing our paper:
````
@article{lun2025towards,
  title={Towards Effective and Sparse Adversarial Attack on Spiking Neural Networks via Breaking Invisible Surrogate Gradients},
  author={Lun, Li and Feng, Kunyu and Ni, Qinglong and Liang, Ling and Wang, Yuan and Li, Ying and Yu, Dunshan and Cui, Xiaoxin},
  journal={arXiv preprint arXiv:2503.03272},
  year={2025}
}
````
