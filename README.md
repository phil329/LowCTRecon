# Medical Image Process Course Homework

## Low Dose CT Image Reconstruction Dataset

Original dataset is available : [https://www.kaggle.com/datasets/andrewmvd/ct-low-dose-reconstruction?resource=download](https://www.kaggle.com/datasets/andrewmvd/ct-low-dose-reconstruction?resource=download)

Here We choose the preprocessed images at :  [https://share.weiyun.com/tWTsIAde](https://share.weiyun.com/tWTsIAde) passwd: dry8pe

## Prerequisites

all in **environment.yaml**

## Training Command

```shell
CUDA_VISIBLE_DEVICES=3 python main.py \
    --data_root=./CT_Reconstruction_256x256_1m \
    --print_freq=10 --save_freq=100 --img_size=256 \
    --epochs=100 --lr=1e-4 --num_workers=32 --device=cuda \
    --model_name=unet4 --batch=16 --run_name=noflip_L2_0427
```
