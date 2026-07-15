# Final Results Report

Auto-generated from real saved artifacts under `outputs/` only. Any section whose backing artifact does not exist yet renders an explicit "not yet generated" message with the command that would produce it, rather than a fabricated number. Regenerate with `python scripts/generate_final_report.py` after (re-)running the relevant experiment/evaluation/robustness scripts.

**Scope note.** This is a research/education artifact. Dataset gender-label predictions reflect labels defined by the source dataset's documentation, not a determination of gender identity, and this system must not be used for employment, policing, surveillance, identity verification, medical diagnosis, admissions, insurance, or other high-impact decisions.

## Architecture Ablation Table

| experiment | backbone_name | backbone_params | adapter_params | total_params | age_mae | gender_accuracy | interval_coverage | mean_epoch_time_seconds |
|---|---|---|---|---|---|---|---|---|
| exp_b_shared_no_adapters | custom_resnet18 | 11176512 | 0 | 11308485 | 5.82343053817749 | 0.9645560908465244 | 0.7420646268229911 | 37.09048658013344 |
| exp_0b_plain_deep18_no_skip_shared_adapters_learned_balance | plain_deep18_no_skip | 11002688 | 525824 | 11660487 | 5.848586559295654 | 0.9334757400061031 | 0.7251930225907921 | 39.895229208058325 |
| exp_f_pretrained_vs_scratch | custom_resnet18 | 11176512 | 525824 | 11834311 | 8.16634464263916 | 0.94 | 0.8175579067772376 | 36.98498457670212 |
| exp_c_shared_adapters | custom_resnet18 | 11176512 | 525824 | 11834309 | 5.455984115600586 | 0.9638101741208603 | 0.7520732056048041 | 36.93944086432457 |
| exp_0_simple_cnn_shared_adapters_learned_balance | simple_cnn | 1569568 | 525824 | 2227367 | 5.827796459197998 | 0.9634507737899243 | 0.7946811552759508 | 37.54421873092652 |
| exp_a_separate | custom_resnet18 | 22353024 | 0 | 22484997 | 5.668204307556152 | 0.9577879581151832 | 0.7606519874177867 | 39.63212828636169 |
| exp_d_shared_adapters_learned_balance | custom_resnet18 | 11176512 | 525824 | 11834311 | 5.821371555328369 | 0.9620045856534556 | 0.7377752359164998 | 37.20003439188004 |
| exp_0c_custom_resnet18_no_zero_init_shared_adapters_learned_balance | custom_resnet18 | 11176512 | 525824 | 11834311 | 5.521562099456787 | 0.9464396284829721 | 0.7357735201601373 | 39.47668685913086 |

## Backbone Comparison (SimpleCNN / PlainDeep18NoSkip / Custom ResNet-18)

| metric | simple_cnn | plain_deep18_no_skip | custom_resnet18 |
|---|---|---|---|
| Backbone | simple_cnn | plain_deep18_no_skip | custom_resnet18 |
| Total parameters | 2227367 | 11660487 | 11834311 |
| Backbone parameters | 1569568 | 11002688 | 11176512 |
| Mean epoch time (s) | 37.54421873092652 | 39.895229208058325 | 37.20003439188004 |
| Inference latency per image (ms) | 1.9772798632566404 | 2.1490122817194663 | 1.9422798522172535 |
| Age MAE | 5.827796459197998 | 5.848586559295654 | 5.821371555328369 |
| Age RMSE | 8.505329132080078 | 8.600918769836426 | 8.282354354858398 |
| Gender-label accuracy | 0.9634507737899243 | 0.9334757400061031 | 0.9620045856534556 |
| Abstention rate | 0.1315413211323992 | 0.06291106662853875 | 0.12696597083214184 |
| Raw interval coverage | 0.7946811552759508 | 0.7251930225907921 | 0.7377752359164998 |
| Calibrated interval coverage | 0.8970546182442093 | 0.9004861309694023 | 0.9027738061195311 |
| Mean interval width | 17.48069953918457 | 14.819450378417969 | 16.322696685791016 |

### SimpleCNN vs Custom ResNet-18 (efficiency/accuracy trade-off, *not* a residual-connection ablation)

SimpleCNN also differs from Custom ResNet-18 in depth and width, not just the presence of residual connections -- any difference below reflects that whole bundle of architectural choices, not residual connections in isolation.

The ResNet experiment achieved a lower age MAE by 0.01 compared with the plain CNN, while using 9,606,944 additional parameters and 0.04 fewer milliseconds per image. This reflects one training run on one dataset/split; it does not, by itself, establish a general causal claim about residual connections.

### PlainDeep18NoSkip vs Custom ResNet-18 (the residual-connection ablation)

PlainDeep18NoSkip matches Custom ResNet-18's stem, stage widths, block layout, embedding size, adapters, loss balancing, and training setup exactly, removing only the residual/skip-connection additions (plus the handful of 1x1 downsample-shortcut projections ResNet has and this backbone structurally cannot) -- this is the controlled comparison that actually isolates what residual connections contribute here.

The ResNet experiment achieved a lower age MAE by 0.03 compared with the plain CNN, while using 173,824 additional parameters and 0.21 fewer milliseconds per image. This reflects one training run on one dataset/split; it does not, by itself, establish a general causal claim about residual connections.

### Custom ResNet-18 vs Custom ResNet-18 (no zero-init residual) -- zero-init ablation

Same architecture, seeds, and training setup as Custom ResNet-18, with `model.backbone.zero_init_residual=false` -- isolates the effect of zero-initializing each residual branch's final normalization layer (a common ResNet training trick) specifically, separate from the presence of the residual connections themselves. See `docs/experiment_plan.md` for why PlainDeep18NoSkip vs. this variant tests residual shortcuts more cleanly than PlainDeep18NoSkip vs. the default (zero-init) ResNet.

The ResNet experiment achieved a lower age MAE by 0.30 compared with the plain CNN, while using 0 fewer parameters and 0.14 additional milliseconds per image. This reflects one training run on one dataset/split; it does not, by itself, establish a general causal claim about residual connections.

## Selective-Risk (AURC) Comparison and Final Interpretation

_Not yet generated. Run `python scripts/compare_backbones.py --checkpoint NAME=path ... --resnet-name <resnet>` to produce this section._

## Mean +/- Std Across Seeds

_Not yet generated. Run `python scripts/run_seeds.py --experiment <name> --seeds 42,123,2026` to produce this section._

## Uncertainty Evaluation

**Important caveat: marginal coverage is not conditional coverage.** Conformal calibration (when used) targets *marginal* coverage -- averaged across the entire test set -- not coverage conditioned on age bucket, gender-label subgroup, or any other subpopulation. A bucket can be systematically under- or over-covered even while the overall test-set coverage exactly matches the target. The per-bucket tables and plots below exist specifically so this can be checked, not assumed away.

Primary model shown below: `exp_d_shared_adapters_learned_balance`.

### Age MAE / Coverage / Width by Age Bucket (raw)

| age_bucket | count | mae | coverage | mean_width | median_width |
|---|---|---|---|---|---|
| 0-10 | 460 | 2.690131425857544 | 0.6847826086956522 | 5.982937812805176 | 3.7007932662963867 |
| 10-20 | 217 | 7.1470842361450195 | 0.5161290322580645 | 15.43061351776123 | 13.9144926071167 |
| 20-30 | 1089 | 4.346002101898193 | 0.7851239669421488 | 14.20637321472168 | 13.046167373657227 |
| 30-40 | 645 | 5.110918045043945 | 0.8217054263565892 | 17.237529754638672 | 16.752933502197266 |
| 40-50 | 359 | 7.456788539886475 | 0.7047353760445683 | 20.20888328552246 | 19.95269775390625 |
| 50-60 | 339 | 8.268611907958984 | 0.7669616519174042 | 22.35731315612793 | 21.53091812133789 |
| 60-70 | 197 | 8.590485572814941 | 0.7461928934010152 | 23.730602264404297 | 23.092960357666016 |
| 70-80 | 91 | 11.770419120788574 | 0.6263736263736264 | 24.707862854003906 | 24.345352172851562 |
| 80-120+ | 100 | 12.961402893066406 | 0.51 | 26.334672927856445 | 25.651071548461914 |

### Age MAE / Coverage / Width by Age Bucket (after conformal calibration)

| age_bucket | count | mae | coverage | mean_width | median_width |
|---|---|---|---|---|---|
| 0-10 | 460 | 2.690131425857544 | 0.941304347826087 | 12.9891996383667 | 10.70705509185791 |
| 10-20 | 217 | 7.1470842361450195 | 0.7741935483870968 | 22.436874389648438 | 20.92075538635254 |
| 20-30 | 1089 | 4.346002101898193 | 0.9494949494949495 | 21.212635040283203 | 20.05242919921875 |
| 30-40 | 645 | 5.110918045043945 | 0.9534883720930233 | 24.243791580200195 | 23.759197235107422 |
| 40-50 | 359 | 7.456788539886475 | 0.8607242339832869 | 27.21514320373535 | 26.958961486816406 |
| 50-60 | 339 | 8.268611907958984 | 0.8790560471976401 | 29.363574981689453 | 28.53717803955078 |
| 60-70 | 197 | 8.590485572814941 | 0.8375634517766497 | 30.73686408996582 | 30.099220275878906 |
| 70-80 | 91 | 11.770419120788574 | 0.7472527472527473 | 31.7141170501709 | 31.351612091064453 |
| 80-120+ | 100 | 12.961402893066406 | 0.67 | 33.34092712402344 | 32.65733337402344 |

![Empirical interval coverage by age bucket](../plots/exp_d_shared_adapters_learned_balance_test_metrics_interval_coverage.png)

![Interval width by age bucket](../plots/exp_d_shared_adapters_learned_balance_test_metrics_interval_width_by_bucket.png)

![Coverage-width trade-off before/after conformal calibration](../plots/exp_d_shared_adapters_learned_balance_test_metrics_coverage_width_tradeoff.png)

### Narrowest and Widest Prediction Intervals

**Narrowest**

| image_path | true_age | q10 | q50 | q90 | width |
|---|---|---|---|---|---|
| data/raw/UTKFace/1_0_0_20161219204759412.jpg.chip.jpg | 1.0 | 0.2372746467590332 | 0.25929537415504456 | 1.3031281232833862 | 1.065853476524353 |
| data/raw/UTKFace/1_1_1_20170109190848182.jpg.chip.jpg | 1.0 | 0.3550635278224945 | 0.3754778504371643 | 1.4894418716430664 | 1.1343783140182495 |
| data/raw/UTKFace/1_1_3_20161220220126609.jpg.chip.jpg | 1.0 | 0.3724203109741211 | 0.42094916105270386 | 1.5217931270599365 | 1.1493728160858154 |
| data/raw/UTKFace/1_0_2_20161219142006881.jpg.chip.jpg | 1.0 | 0.5193575024604797 | 0.5840781331062317 | 1.6914026737213135 | 1.1720452308654785 |
| data/raw/UTKFace/2_1_2_20161219140840080.jpg.chip.jpg | 2.0 | 0.4575585722923279 | 0.5051780343055725 | 1.6361498832702637 | 1.178591251373291 |

**Widest**

| image_path | true_age | q10 | q50 | q90 | width |
|---|---|---|---|---|---|
| data/raw/UTKFace/50_1_0_20170110143400936.jpg.chip.jpg | 50.0 | 49.09925842285156 | 67.89872741699219 | 88.07933807373047 | 38.980079650878906 |
| data/raw/UTKFace/55_1_0_20170117155022829.jpg.chip.jpg | 55.0 | 26.152334213256836 | 43.706146240234375 | 64.5798110961914 | 38.42747497558594 |
| data/raw/UTKFace/85_1_0_20170110183536244.jpg.chip.jpg | 85.0 | 44.03086853027344 | 62.362091064453125 | 82.26776123046875 | 38.23689270019531 |
| data/raw/UTKFace/31_1_0_20170117133158992.jpg.chip.jpg | 31.0 | 36.720848083496094 | 54.30280303955078 | 74.8665542602539 | 38.14570617675781 |
| data/raw/UTKFace/61_1_0_20170120134639935.jpg.chip.jpg | 61.0 | 46.77733612060547 | 65.06438446044922 | 84.89832305908203 | 38.12098693847656 |

## Robustness Degradation

_Pairwise robustness comparison not yet generated. Run `python scripts/run_robustness.py --checkpoint <checkpoint>` for each model, then `python scripts/compare_backbones.py --robustness-csv NAME=path/to/robustness_results.csv ...` for all of them together._

### agegender_runs

**Clean baseline**

| corruption | severity | param | n_samples | age_mae | age_rmse | interval_coverage | mean_interval_width | interval_coverage_calibrated | mean_interval_width_calibrated | gender_accuracy | abstention_rate | mean_confidence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clean | 0 | nan | 300 | 5.753019027709961 | 8.097393844316231 | 0.77 | 16.319618225097656 | 0.8833333333333333 | 23.32588195800781 | 0.9621212121212122 | 0.12 | 0.9409754276275636 |

**Mean metrics by corruption type (across severities)**

| corruption | age_mae | gender_accuracy | abstention_rate | mean_confidence | mean_interval_width | interval_coverage_calibrated | mean_interval_width_calibrated |
|---|---|---|---|---|---|---|---|
| gaussian_blur | 6.52653082460165 | 0.9533015244633726 | 0.11999999999999995 | 0.9415187239646912 | 16.291828155517578 | 0.85 | 23.2980899810791 |
| gaussian_noise | 29.084703595969415 | 0.9292950436616287 | 0.18666666666666662 | 0.9105086525281271 | 27.325612386067707 | 0.37777777777777777 | 34.331874211629234 |
| grayscale | 6.05762463927269 | 0.9496722910326559 | 0.15777777777777774 | 0.9242851734161377 | 17.983917872111004 | 0.8977777777777778 | 24.99018096923828 |
| high_brightness | 7.26937752455473 | 0.9238357821953328 | 0.13666666666666663 | 0.9306780298550924 | 18.90007146199544 | 0.8555555555555555 | 25.906333287556965 |
| high_contrast | 6.277866950117879 | 0.9436158196189979 | 0.13222222222222216 | 0.9389949242273966 | 16.110552151997883 | 0.861111111111111 | 23.11681365966797 |
| jpeg_compression | 6.4956553252538045 | 0.9570274993278796 | 0.1211111111111111 | 0.9409494996070862 | 17.7748597462972 | 0.8811111111111112 | 24.7811222076416 |
| low_brightness | 7.4871183747715415 | 0.9354412737888266 | 0.18333333333333332 | 0.9091514746348063 | 19.011380513509113 | 0.86 | 26.017642974853516 |
| low_contrast | 7.387898050281737 | 0.9265514002477747 | 0.2111111111111111 | 0.9007585644721985 | 23.137949625651043 | 0.8977777777777778 | 30.144211451212566 |
| low_resolution | 7.172088145812353 | 0.9458012453473882 | 0.11777777777777772 | 0.938500702381134 | 16.47890027364095 | 0.8244444444444444 | 23.485161463419598 |
| partial_crop | 8.638189301623237 | 0.8749584224281302 | 0.19222222222222216 | 0.9003905256589254 | 21.071678161621094 | 0.8288888888888889 | 28.07793935139974 |
| partial_occlusion | 10.387502223451934 | 0.7948580129213116 | 0.1233333333333333 | 0.9343084295590719 | 16.494507789611816 | 0.6744444444444445 | 23.500771204630535 |

![agegender_runs: robustness curve (age_mae)](../robustness/robustness_age_mae.png)

![agegender_runs: degradation vs. severity (age_mae % change)](../robustness/degradation_age_mae_pct_change.png)

![agegender_runs: robustness curve (gender_accuracy)](../robustness/robustness_gender_accuracy.png)

![agegender_runs: degradation vs. severity (gender_accuracy % change)](../robustness/degradation_gender_accuracy_pct_change.png)

![agegender_runs: robustness curve (abstention_rate)](../robustness/robustness_abstention_rate.png)

![agegender_runs: degradation vs. severity (abstention_rate % change)](../robustness/degradation_abstention_rate_pct_change.png)

## Parameter Count and Inference Latency Comparison

![Parameter count vs inference latency per experiment](../plots/final_report/parameter_latency_comparison.png)

| experiment | total_parameters | latency_ms_per_image |
|---|---|---|
| exp_b_shared_no_adapters | 11308485 | 1.935322284016706 |
| exp_0b_plain_deep18_no_skip_shared_adapters_learned_balance | 11660487 | 2.1490122817194663 |
| exp_f_pretrained_vs_scratch | 11834311 | 1.981772931944073 |
| exp_c_shared_adapters | 11834309 | 1.8270812717750817 |
| exp_0_simple_cnn_shared_adapters_learned_balance | 2227367 | 1.9772798632566404 |
| exp_a_separate | 22484997 | 2.4271965163211533 |
| exp_d_shared_adapters_learned_balance | 11834311 | 1.9422798522172535 |
| exp_0c_custom_resnet18_no_zero_init_shared_adapters_learned_balance | 11834311 | 2.0795735012029355 |

## Evidence-Based Findings

- **Efficiency/accuracy trade-off (SimpleCNN vs. Custom ResNet-18, *not* a residual-connection ablation -- depth and width differ too):** The ResNet experiment achieved a lower age MAE by 0.01 compared with the plain CNN, while using 9,606,944 additional parameters and 0.04 fewer milliseconds per image. This reflects one training run on one dataset/split; it does not, by itself, establish a general causal claim about residual connections.

- **Residual-connection ablation (PlainDeep18NoSkip vs. Custom ResNet-18, depth/width held fixed):** The ResNet experiment achieved a lower age MAE by 0.03 compared with the plain CNN, while using 173,824 additional parameters and 0.21 fewer milliseconds per image. This reflects one training run on one dataset/split; it does not, by itself, establish a general causal claim about residual connections.

- Under the measured corruptions, age MAE degraded from 5.75 years (clean) to as much as 47.58 years under 'gaussian_noise' at severity 3, an increase of 41.82 years.
