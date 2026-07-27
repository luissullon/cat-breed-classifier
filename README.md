# Cat Breed Classifier

A fine-grained image classification project that identifies **12 cat breeds** from photographs, built as an end-to-end comparison of classical feature engineering, CNNs, and transfer learning (ResNet, EfficientNet, ViT), tuned with Bayesian hyperparameter optimization.

Runs for free on **Google Colab** (GPU runtime).

## Results

| Metric (held-out test set) | Score |
|---|---|
| Accuracy | 93.3% |
| Macro-F1 | 93.4% |
| Top-3 accuracy | 99.2% |
| Test loss | 0.317 |

**Final model:** EfficientNet-B0, fine-tuned with Optuna-selected hyperparameters (`lr_head ≈ 1.29e-3`, `lr_backbone ≈ 4.96e-4`, `weight_decay ≈ 4.53e-4`, `drop_rate ≈ 0.30`, `batch_size = 16`).

### Model comparison (validation macro-F1)

| Model | Macro-F1 | Params | Latency (ms/img) |
|---|---|---|---|
| MLP (engineered features) | 26.9% | — | — |
| CNN (from scratch) | 62.0% | 1.18M | 2.1 |
| ResNet-50 (transfer) | 92.4% | 23.5M | 5.7 |
| EfficientNet-B0 (transfer, untuned) | 85.6% | 4.0M | 8.3 |
| ViT-Small (transfer) | 93.9% | 21.7M | 5.5 |
| **EfficientNet-B0 (tuned, test set)** | **93.4%** | 4.0M | 8.2 |

## Dataset

[Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/), filtered to its 12 cat breeds: Abyssinian, Bengal, Birman, Bombay, British Shorthair, Egyptian Mau, Maine Coon, Persian, Ragdoll, Russian Blue, Siamese, and Sphynx (~2,400 images).

## Methodology

1. **Data splitting** — stratified 70/15/15 train/validation/test split, done before any feature computation or augmentation to avoid leakage. The test set is touched exactly once, at final evaluation.
2. **Feature engineering** — hand-crafted CV features (HSV color histograms, LBP texture, Canny edge density) feed a Multi-Layer Perceptron baseline, quantifying how much signal is available without deep learning.
3. **Modeling** — a CNN trained from scratch, then transfer learning with ResNet-50, EfficientNet-B0, and ViT-Small (ImageNet-pretrained), using partial-layer unfreezing, discriminative learning rates, mixed-precision training, and early stopping.
4. **Hyperparameter optimization** — Optuna Bayesian search (TPE sampler with median pruning) over learning rates, weight decay, dropout, and batch size, evaluated on the validation set only.
5. **Metric selection** — macro-F1 is the primary/model-selection metric (robust to class imbalance and per-breed performance gaps), with accuracy and top-3 accuracy reported as secondary metrics.
6. **Interpretability & error analysis** — confusion matrix, misclassification gallery, and Grad-CAM saliency maps to inspect what the final model attends to.

## Tech stack

`PyTorch` · `torchvision` · `timm` (EfficientNet, ViT) · `Optuna` · `scikit-learn` · `scikit-image` / `OpenCV` (feature engineering) · `pytorch-grad-cam` · `pandas` / `matplotlib` / `seaborn`

## Project structure

```
.
├── cat_breed_classification.ipynb   # Full pipeline: data → features → models → tuning → evaluation
├── artifacts/
│   ├── efficientnet_b0_cat_breed_final.pth   # Final trained model weights
│   ├── model_comparison.csv                  # Macro-F1 / params / latency per model
│   ├── best_hyperparameters.json             # Optuna-selected hyperparameters
│   └── test_metrics.json                     # Held-out test metrics
└── README.md
```

## Running it

1. Open `cat_breed_classification.ipynb` in Google Colab.
2. `Runtime → Change runtime type → GPU` (a free T4 is enough).
3. `Runtime → Run all`. Total runtime: ~30-45 minutes, including the Optuna search.

The notebook downloads the dataset directly from its source, so no manual data setup is required.

## Key findings

- Hand-crafted color/texture features alone reach only 26.9% macro-F1 — well above chance (~8.3% for 12 classes) but far from usable, showing breed differences are too fine-grained for low-dimensional global image statistics.
- ImageNet pretraining accounts for a >30-point macro-F1 jump over training a comparable CNN from scratch (62.0% → 85%+).
- Hyperparameter tuning closed EfficientNet-B0's gap to the other transfer models, taking it from 85.6% to 93.4% macro-F1 while using ~5x fewer parameters than ResNet-50 or ViT-Small.
- ViT-Small led all models on validation (93.9%) without ever being tuned, suggesting further headroom with the same optimization budget.

## Future work

- Run the Optuna search over ViT-Small.
- Ensemble ResNet-50 / EfficientNet-B0 / ViT-Small, which land in a tight 92-94% band and likely make partially independent errors.
- Target the most-confused breed pairs (from the confusion matrix) with additional data or face-cropping.
- Package inference behind a FastAPI/Gradio demo.
