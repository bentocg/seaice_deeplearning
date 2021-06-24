# Fine-scale sea ice extraction for high-resolution satellite imagery with weakly-supervised CNNs
> Fully-automated pipeline to extract ice floes from WV03 panchromatic imagery with a U-Net. Trained on a small set of hand-labelled Antarctic pack-ice and background images and a much larger set of weakly-labelled pack-ice images obtained with a watershed segmentation algorithm. Best results are obtained using test-time-augmentation. Model predictions are largely robust to context, adding more flexibility in applications when compared with threshold-based methods typically employed in sea ice segmentation.

---
### Highlights:
* Leverages fine-tuning from synthetic data to greatly improve out-of-sample performance.
* \> 0.85 F1 score in a non-trivial, hand-annotated test set.
* Over 30% improvement when compared with threshold-based methods.
* Best model weights (incoming) are easily loaded with the PyTorch Segmentation Models package.
* Over 850 random-search experiments ran for hyperparameter tuning with the Bridges2 supercomputer.

---
### Contents:
* Training script.
* Model evaluation script.
* Prediction script (incoming).
* Saved model weights (incoming).
* Dataset classes.
* Several implementations of Semantic Segmentation loss functions.
