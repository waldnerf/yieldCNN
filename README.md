# Lean Temporal Convolutional Neural Network for yield forecasting
Training temporal Convolution Neural Networks (CNNs) on satellite image time series for yield forecasting.


## Prerequisites
To set up the environment:

```
git clone https://github.com/waldnerf/yieldCNN.git
cd yieldCNN
conda env create -f environment.yml
conda activate leanyf
```
If a GPU is available, then type in:
```pip install tensorflow-gpu==2.3.0```

You can monitor GPU usage with the following command: 
```watch -n 1 nvidia-smi```

## Contributors
 - [Dr. Franz Waldner](https://scholar.google.com/citations?user=4z2zcXwAAAAJ&hl=en&oi=ao)


