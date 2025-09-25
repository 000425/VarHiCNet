## VarHiCNet
VarHiCNet is a structural variation detection method based on Hi-C and neural networks

## Requirements
cython，scipy，Python3.9，PyTorch 1.4+，and，torchvision 0.5+，numpy, pandas, Matplotlib, cooler, seaborn, sklearn

## plot contact matrix image
plot inter-chromosomal contact matrix image
```python
python plot_inter_img.py
```
plot intra-chromosomal contact matrix image
```python
python plot_intra_img.py
```
## Creating a dataset
```python
python whj_helas3_gen.py
```

## SV calling by VarHiCNet
```python
CUDA_VISIBLE_DEVICES=0 python /mnt/sdc/wanghaojie//main.py --batch_size 8 --output_dir /mnt/sdc/wanghaojie/newstart/111111 --coco_path /mnt/sdc/wanghaojie/newstart/RT-DETR-main/rtdetr_pytorch/data --resume /mnt/sdc/wanghaojie/newstart/detr_r50_2.pth --epochs 200 --use_feature_fusion --backbone resnet50 --use_aspp
```

```python
python /mnt/sdc/wanghaojie/newstart/detr-main/新predict.py
```
