# Loss Diagnostics Report

- **Status:** `overfit`
- **n_classes:** 4
- **epochs:** 12

## Curve stats
- train_loss: start=1.3561 end=0.2985 drop_ratio=0.7799
- val_loss:   min=0.6366 end=0.7942 rebound_ratio=0.2477
- train_acc_end=0.9126  val_acc_end=0.6165  val_acc_mean=0.5222

## Flags
- val_loss rebounded >15% after its minimum while training improved → overfitting signal.

## Next actions
- Use early stopping (pick epoch at min val_loss).
- Add regularization (dropout/L2), reduce epochs, or simplify model.
- Get more diverse training data / augment data.

## Raw history (first 5 epochs)
```json
{
  "train_loss": [
    1.3560943415950886,
    1.111677221568768,
    0.9693930791588963,
    0.8273852576406561,
    0.6501740531675891
  ],
  "val_loss": [
    1.3519809209268365,
    1.2009454460917397,
    1.0286836015063279,
    0.8617573327767933,
    0.7926101128312829
  ],
  "train_acc": [
    0.34571672177836893,
    0.4588655975867689,
    0.5597246633134371,
    0.6333912262653443,
    0.6955433683632825
  ],
  "val_acc": [
    0.29829078813517684,
    0.3572214272539862,
    0.41567233183710234,
    0.4863360147277505,
    0.528194278468325
  ]
}
```
