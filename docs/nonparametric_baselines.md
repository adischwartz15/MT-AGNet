
# Non-Parametric Baselines (Raw/PCA and Frozen-Backbone)


## The two feature pipelines

- **`raw_pca`** - flattened raw pixels (resized to a small, fixed size) ->
  train-only `StandardScaler` -> train-only PCA -> optional L2-normalization.
  The "no learning at all" floor.
- **`frozen_backbone`** - pooled features from a frozen, ImageNet-pretrained
  backbone (`src/models/pretrained_resnet.py`, adapters/heads never
  attached, weights never fine-tuned) -> the same scaler/PCA/L2-norm
  protocol. Tests whether generic pretrained visual features alone (with no
  task-specific training at all) already carry most of the signal.

Implemented in `src/evaluation/nonparametric/`:

```
features.py    extract_raw_pixel_features, extract_frozen_backbone_features
pipeline.py    fit_feature_pipeline (train-only scaler/PCA), tune_knn_age,
               tune_knn_gender, tune_kernel_regression_age, tune_kde_gender
kernels.py     NadarayaWatsonRegressor, ClassConditionalKDEClassifier
               (numerically safe: NN fallback on kernel-weight underflow,
               diagonal-covariance fallback for tiny/singular classes)
```

## Protocol: which split does what

| Split | Used for |
|---|---|
| **train** | Fitting the scaler, PCA, and the reference set for k-NN/kernel methods -- never validation, calibration, or test. |
| **validation** | Selecting every hyperparameter (k, distance metric, PCA dimensionality, L2-normalization, kernel bandwidth) -- never test. |
| **calibration** | Fitting split-conformal calibration for the k-NN age-interval baseline only -- never validation or test. |
| **test** | Final, one-shot reported numbers -- never used for any selection. |

This is the same 4-way split discipline every other evaluation path in this
project follows (see [docs/reproducibility.md](reproducibility.md#stratified-locked-split)).
PCA is intentionally restricted to a small number of components
(`safe_n_components`, capped by both a fixed grid and by `min(n_samples,
n_features)`) - the kernel/KDE methods run in this reduced space, never on
full-dimensional raw pixels or full-width backbone features, since
Nadaraya-Watson/KDE degrade badly in high dimensions.


