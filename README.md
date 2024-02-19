# Uncertainty-based Extensible-codebook Federated Learning


## Abstract
Federated learning, aimed at leveraging vast datasets distributed across numerous locations and devices, confronts a crucial challenge: the heterogeneity of data across different silos. While previous studies have explored discrete representations to enhance model generalization across minor distributional shifts, these approaches often struggle to adapt to new data silos with significantly divergent distributions. In response, we have identified that models derived from federated training exhibit markedly increased uncertainty when applied to data silos with unfamiliar distributions. Consequently, we propose an innovative yet straightforward iterative framework, termed 'Uncertainty-Based Extensible-Codebook Federated Learning (UEFL)'. This framework dynamically maps latent features to trainable discrete vectors, assesses the uncertainty, and specifically extends the discretization dictionary or codebook for silos exhibiting high uncertainty. Our approach aims to simultaneously enhance accuracy and reduce uncertainty by explicitly addressing the diversity of data distributions, all while maintaining minimal computational overhead in environments characterized by heterogeneous data silos. Through experiments conducted on five well-known datasets, our method has demonstrated its superiority, achieving significant improvements in accuracy (by 3\%--22.1\%) and uncertainty reduction (by 38.83\%--96.24\%), thereby outperforming contemporary state-of-the-art methods.

![image](https://github.com/destiny301/uefl/blob/main/flowchart.png)

## Updates

## Data
Prepare the data as the following structure:
```shell
datasets/
├──MNIST/
├──fmnist/
│  ├── train-images-idx3-ubyte
│  ├── ......
├──.....
```

## Simple Start
### MNIST with 9 silos
```shell
python train.py --data mnist --num_silo 9 --num_dist 3 --sample 2000 --enocder cnn --depth 3 --num_codes 64 --seg 1 --round 20 --epoch 20 --step 20 --thd 0.1 --workdir /your/save/folder
```
To obtain the best performance, please use pretrained model on large image dataset

## Citation
If you use UEFL in your research or wish to refer to the results published here, please use the following BibTeX entry. Sincerely appreciate it!
