# Dataset Summary

## Tabular climate dataset

Input: `Tabular processing/final_climate_lag_dataset.csv`. Outputs: `data/climate_train.csv`, `data/climate_val.csv`, and `data/climate_test.csv`. Target: `risk_label` with Low, Medium, and High values. Features include rainfall, temperature, humidity, NDVI, lag features, and `city_encoded`.

## ResNet-18 image dataset

The image branch uses image files only for ResNet-18 binary classification; YOLO object detection is not used. YOLO label folders have been removed. Dataset location: `Image processing/Res-18 Image processing/MosquitoFusion Dataset/`, with `train/images/`, `valid/images/`, and `test/images/` splits. Binary `Breeding` / `NonBreeding` labels are stored in `data/yolo_train_split.csv`, `data/yolo_val_split.csv`, and `data/yolo_test_split.csv`. The original YOLO class mapping was used to derive the binary labels but is no longer used directly.

## Combined system

The tabular model predicts climate-only disease risk. The image model uses ResNet-18 for Breeding / NonBreeding image classification. Future fusion can combine climate features with image features for final risk prediction.
