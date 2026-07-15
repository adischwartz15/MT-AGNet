# Comprehensive Backbone Comparison Suite

Practical guide to `scripts/compare_backbones.py`. For the methodology
behind each analysis (selective prediction, AURC, paired bootstrap,
tail-error analysis, the conditional "is residual complexity justified"
interpretation), see `docs/architecture_analysis.md` (section 9). For the
experiments being compared and why each pairing matters, see
`docs/experiment_plan.md`.

## What it does

Post-hoc analysis across two or more already-trained checkpoints --
**never retrains**, only re-runs inference (a single forward pass per
test-set image) against each checkpoint's own test split.

```bash
python scripts/compare_backbones.py \
    --checkpoint simple_cnn=experiments/exp_0_simple_cnn_shared_adapters_learned_balance/seed_42/checkpoints/exp_0_simple_cnn_shared_adapters_learned_balance_best_balanced_score.pt \
    --checkpoint plain_deep18_no_skip=experiments/exp_0b_plain_deep18_no_skip_shared_adapters_learned_balance/seed_42/checkpoints/exp_0b_plain_deep18_no_skip_shared_adapters_learned_balance_best_balanced_score.pt \
    --checkpoint custom_resnet18=experiments/exp_d_shared_adapters_learned_balance/seed_42/checkpoints/exp_d_shared_adapters_learned_balance_best_balanced_score.pt \
    --calibration-dir simple_cnn=experiments/exp_0_simple_cnn_shared_adapters_learned_balance/seed_42/calibration \
    --calibration-dir plain_deep18_no_skip=experiments/exp_0b_plain_deep18_no_skip_shared_adapters_learned_balance/seed_42/calibration \
    --calibration-dir custom_resnet18=experiments/exp_d_shared_adapters_learned_balance/seed_42/calibration \
    --robustness-csv simple_cnn=experiments/exp_0_simple_cnn_shared_adapters_learned_balance/seed_42/robustness/robustness_results.csv \
    --robustness-csv custom_resnet18=experiments/exp_d_shared_adapters_learned_balance/seed_42/robustness/robustness_results.csv \
    --resnet-name custom_resnet18 \
    --output-dir outputs/backbone_comparison
```

(Optionally add
`--checkpoint custom_resnet18_no_zero_init=experiments/exp_0c_.../seed_42/checkpoints/..._best_balanced_score.pt`
for the zero-init-residual ablation -- see `docs/final_evaluation_protocol.md`.)

A shorter, Makefile-driven form covers the common two-checkpoint case
(`CHECKPOINTS`/`RESNET_NAME` override the defaults shown by
`make -n compare-backbones`):

```bash
make compare-backbones CHECKPOINTS="simple_cnn=checkpoints/exp_0_..._best_balanced_score.pt custom_resnet18=checkpoints/exp_d_..._best_balanced_score.pt" RESNET_NAME=custom_resnet18
```
