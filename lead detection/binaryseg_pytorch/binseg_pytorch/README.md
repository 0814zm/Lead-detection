
#A Shape-Aware Network for Arctic Lead Detection From Sentinel-1 SAR Images


# Abstract

Accurate detection of sea ice leads is essential for safe navigation in polar regions. In this paper, a shape-aware (SA) network, SA-DeepLabv3+, is proposed for automatic lead detection from syn-thetic aperture radar (SAR) images. Considering the fact that training data are limited in the task of lead detection, we construct a dataset fusing dual-polarized (HH, HV) SAR images from the C-band Sentinel-1 satellite. Taking the DeepLabv3+ as the baseline network, we introduce a shape-aware module (SAM) to combine multi-scale semantic features and shape information and therefore better capture the shape characteristics of leads. A squeeze-and-excitation channel-position attention module (SECPAM) is designed to enhance lead feature extraction. Segmentation loss generated by the segmentation network and shape loss generated by the shape-aware stream are combined to optimize the network during training. Postprocessing is performed to filter out segmentation errors based on the aspect ratio of leads. Experimental results show that the proposed method outperforms the existing benchmarking deep learning methods, reaching 96.82% for overall accuracy, 93.01% for F1-score, and 91.48% for MIoU. It is also found that the fusion of dual-polarimetric SAR channels as the input could effectively improve the accuracy of sea ice leads detection.

# Keywords
lead detection; synthetic aperture radar (SAR) images; shape-aware module (SAM); squeeze-and-excitation channel-position attention module (SECPAM)

## Usage visdom
```
启动visdom：python -m visdom.server
```

## Usage

For quick hints about commands:

```
python main.py -h
```

Modify the condiguration in the `settings.py` file

### Training

After customizing the `settings.py`, use the following command to start training
```
python main.py --cuda train
```
### Evaluation
For evaluation, put all your test images in a folder and set path in the `settings.py`. Then run the following command:
```
python main.py --cuda eval
```
The results will be place in the `results` directory


