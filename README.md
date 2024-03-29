# MCOR

## Overview
Images obtained from different modalities can effectively enhance the accuracy and reliability of the detection model by complementing specialized information from visible (RGB) and infrared (IR) images.  
However, integrating information from multiple modalities faces the following challenges:  
1) distinct characteristics of RGB and IR images lead to the problem of modality imbalance
2) fusing multimodal information can greatly affect the detection accuracy, as some of the unique information provided by each modality is lost during the integration process
3) RGB and IR images are fused while preserving the noise of each modality.

To address these issues, we propose a novel multispectral object detection network which contains two main components; 1) Cross-modal Information Complementary (CIC) module, and 2) Cosine Similarity Channel Resampling (CSCR) module. 
The proposed method addresses the modality imbalance problem and efficiently fuses RGB and IR images in the feature level. 
Extensive experimental results on three different benchmark datasets, LLVIP, FLIR and M<sup>3</sup>FD, verify the effectiveness and generalization performance of the proposed multispectral object detection network compared with other state-of-the-art methods.
### Overall Architecture
![alt text](/figures/over_arch_fusion_final.png)

## Installation 
Python>=3.6.0 is required with all requirements.txt installed including PyTorch>=1.7 (The same as [YOLOv5](https://github.com/ultralytics/yolov5) and [YOLOv8](https://github.com/ultralytics/ultralytics)).

Download repository:
```bash
git clone https://github.com/datu0615/MCOR
```
Create conda environment:
 ```bash
$ conda create -n MCOR python=3.8
$ conda activate MCOR
$ cd  MCOR
$ pip install -r requirements.txt
```

## Datasets and Annotations
[KAIST dataset](https://soonminhwang.github.io/rgbt-ped-detection/), [FLIR dataset](https://www.flir.cn/oem/adas/adas-dataset-form/), [FLIR-aligned dataset](https://github.com/zonaqiu/FLIR-align), [LLVIP dataset](https://bupt-ai-cz.github.io/LLVIP/), [M<sup>3</sup>FD dataset](https://github.com/dlut-dimt/TarDAL)
- Improved KAIST Testing Annotations provided by Liu et al.[Link to download](https://docs.google.com/forms/d/e/1FAIpQLSe65WXae7J_KziHK9cmX_lP_hiDXe7Dsl6uBTRL0AWGML0MZg/viewform?usp=pp_url&entry.1637202210&entry.1381600926&entry.718112205&entry.233811498) 
- Sanitized KAIST Training Annotations provided by Li et al.[Link to download](https://github.com/Li-Chengyang/MSDS-RCNN) 
- Improved KAIST Training Annotations provided by Zhang et al.[Link to download](https://github.com/luzhang16/AR-CNN) 
## Tools
- Evalutaion codes.[Link to download](https://github.com/Li-Chengyang/MSDS-RCNN/tree/master/lib/datasets/KAISTdevkit-matlab-wrapper)
- Annotation: vbb format->xml format.[Link to download](https://github.com/SoonminHwang/rgbt-ped-detection/tree/master/data/scripts)

## Run
#### Change the data cfg
some example in data/multispectral/
#### Change the model cfg
some example in models/MCOR/

### Train Test and Detect
train:  
```bash
python train.py --data data/multispectral/{dataset}.yaml --cfg models/MCOR/{model}.yaml --epochs 100 --batch-size {batch_size} --device {device}
```

test:  
```bash
python test.py --weights runs/train/{model}/weights/best.pt --data data/multispectral/{dataset}.yaml --batch-size {batch_size} --device {device}
```

detect:  
```bash
python detect_twostream.py --weight runs/train/{model}/weights/best.pt --source1 datasets/{dataset}/visible/test --source2 datasets/{dataset}/infrared/test --device {device}
```
<!--
### Demo
**Night Scene**
<div align="left">
<img src="https://github.com/datu0615/MCOR/figures/day_visible.gif" width="600"></img>
<img src="https://github.com/datu0615/MCOR/figures/day_infrared.gif" width="600"></img>
</div>

**Day Scene**
<div align="left">
<img src="https://github.com/datu0615/MCOR/figures/night_visible.gif" width="600">
<img src="https://github.com/datu0615/MCOR/figures/night_infrared.gif" width="600">
</div>
-->

## Results
|Dataset|CIC|CSCR|mAP50|mAP75|mAP|
|:---------:|:------------:|:-----------:|:-----:|:-----------------:|:-------------:|
|FLIR|||74.2|29.1|35.6|
|FLIR|✔️||75.3|32.6|38.5|
|FLIR||✔️|74.6|29.7|36.2|
|FLIR|✔️|✔️|**77.3 (Δ3.1)**|**30.4 (Δ1.3)**|**37.6 (Δ2.0)**|
|LLVIP|||96.7|71.3|62.3|
|LLVIP|✔️||97.2|73.5|64.2|
|LLVIP||✔️|96.7|72.4|63.9|
|LLVIP|✔️|✔️|**97.6 (Δ0.9)**|**73.7 (Δ2.4)**|**64.9 (Δ2.6)**|
|M<sup>3</sup>FD|||82.6|55.8|52.3|
|M<sup>3</sup>FD|✔️||86.1|62.4|57.8|
|M<sup>3</sup>FD||✔️|85.2|60.2|56.6|
|M<sup>3</sup>FD|✔️|✔️|**87.2 (Δ4.6)**|**60.9 (Δ5.1)**|**57.2 (Δ4.9)**|

|Dataset|Method|CSCR|mAP50|mAP75|mAP|
|:---------:|:------------:|:------------:|:-----:|:-----------------:|:-------------:|
|FLIR|Two Stream||74.2|29.1|35.6|
|FLIR|Two Stream|✔️|**74.6 (Δ0.4)**|**29.7 (Δ0.6)**|**36.2 (Δ0.6)**|
|FLIR|CFT||74.5|30.1|36.5|
|FLIR|CFT|✔️|**75.6 (Δ1.1)**|**30.9 (Δ0.8)**|**37.2 (Δ0.7)**|
|LLVIP|Two Stream||96.7|71.3|62.3|
|LLVIP|Two Stream|✔️|**96.8 (Δ0.1)**|**71.4 (Δ0.1)**|**62.8 (Δ0.2)**|
|LLVIP|CFT||96.8|70.7|62.9|
|LLVIP|CFT|✔️|**97.3 (Δ0.5)**|**71.3 (Δ0.6)**|**62.3 (Δ0.6)**|
|M<sup>3</sup>FD|Two Stream||82.6 | 55.8  | 52.3
|M<sup>3</sup>FD|Two Stream|✔️|**85.2 (Δ2.6)**|**60.3(Δ5.5)**|**56.6 (Δ4.3)**|
|M<sup>3</sup>FD|CFT||83.2 | 55.1  | 52.2
|M<sup>3</sup>FD|CFT|✔️|**86.0 (Δ2.8)**|**59.3(Δ4.2)**|**57.2 (Δ5.0)**|


<!--
|Dataset|CFT|mAP50|mAP75|mAP|
|:---------: |------------|:-----:|:-----------------:|:-------------:|
|FLIR||73.0|32.0|37.4|
|FLIR| ✔️ |**78.7 (Δ5.7)**|**35.5 (Δ3.5)**|**40.2 (Δ2.8)**|
|LLVIP||95.8|71.4|62.3|
|LLVIP| ✔️ |**97.5 (Δ1.7)**|**72.9 (Δ1.5)**|**63.6 (Δ1.3)**|
|VEDAI||79.7 | 47.7  | 46.8
|VEDAI| ✔️ |**85.3 (Δ5.6)**|**65.9(Δ18.2)**|**56.0 (Δ9.2)**|


### LLVIP
Log Average Miss Rate 
|Model| Log Average Miss Rate |
|:---------: |:--------------:|
|YOLOv3-RGB|37.70%|
|YOLOv3-IR|17.73%|
|YOLOv5-RGB|22.59%|
|YOLOv5-IR|10.66%|
|Baseline(Ours)|**6.91%**|
|CFT(Ours)|**5.40%**|

Miss Rate - FPPI curve
<div align="left">
<img src="https://github.com/DocF/multispectral-object-detection/blob/main/MR.png" width="500">
</div>
-->

## Acknowlegment
The code is borrowed from [YOLOv5](https://github.com/ultralytics/yolov5) and [CFT](https://github.com/DocF/multispectral-object-detection). Thanks for their contribution.

  
