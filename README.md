# Pneumonia detection on chest X-ray (ResNet50, transfer learning)

Binary classifier (NORMAL vs PNEUMONIA) fine-tuned from ImageNet ResNet50, with a Grad-CAM heatmap and a web demo.

> **Research prototype. Not a medical device. Not for clinical use.** Trained and tested only on a public dataset; no patient data is included in this repository.

**Live demo:** _Hugging Face Space link to add_

## Results (held-out test set, 879 images)

| Metric | Value |
|---|---|
| Accuracy | 93.7% |
| Sensitivity (recall) | 91.7% |
| Specificity | 99.2% |
| Precision | 99.7% |
| F1 | 95.5% |
| AUC-ROC | 0.9925 |

Confusion matrix: TN 236, FP 2, FN 53, TP 588 (`results/metrics.json`). Numbers come from the run that produced the model used in the demo; a previous run with the same settings gave 93.5% accuracy and AUC 0.9925, so expect small run-to-run variation.

## Method

- Data: Kermany et al. 2018, chest X-ray images (Kaggle `paultimothymooney/chest-xray-pneumonia`, CC BY 4.0). The original `val` folder has only 16 images, so all images were pooled and re-split 70/15/15 (stratified by class).
- Model: ResNet50 (ImageNet weights), frozen base + global average pooling + dropout 0.5 + sigmoid head, then fine-tuning of the last ~30 layers (Adam 1e-4, then 1e-5).
- Class imbalance (about 1:2.9) handled with class weights; light augmentation (rotation, zoom, shifts, horizontal flip); early stopping on validation AUC.
- Explainability: Grad-CAM on the last convolutional block.

## Limitations (read before quoting the numbers)

- The split is by image, not by patient. The dataset contains several images per patient, so the test set may share patients with the training set and the metrics are probably optimistic.
- The images come from children aged 1 to 5 in a single hospital system. The model has not been validated on adults or on other scanners.
- At the default 0.5 threshold, 53 of 641 pneumonia cases were missed (sensitivity 91.7%). A screening use would need a lower threshold and external validation.
- Grad-CAM shows where the network looked, not that it found a lesion.

## Run it

Train (Google Colab, free T4 GPU): open `pneumonia_resnet50_training.ipynb`, run all cells; the last cell saves `pneumonia_resnet50.keras`.

Demo locally:

```bash
cd demo
pip install -r requirements.txt
# put pneumonia_resnet50.keras in this folder
python app.py
```

## Author

Abdullah Ghani, data and AI engineering student at ECE Paris. Developed as part of a master's thesis project on computer vision in medical imaging.
