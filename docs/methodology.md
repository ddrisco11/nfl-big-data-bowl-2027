# Methods and reproducibility

## Data and population

The competition files are `players.csv`, `combine_results.csv`, `combine_tracking.csv`, `player_play.csv`, `games.csv`, `player_career_successes.csv`, and the three `game_tracking_YYYY.csv` files for 2023–2025. Career outcomes are not used in the route-recognition target.

The nine-file release contains 510 prospects and 22.65 million game-tracking rows. This is cohort tracking, not full-roster tracking. Absent players and ball positions are never reconstructed. Both `NA` and empty cells are read as missing values. Nullified plays, invalid play directions, and ambiguous repeated snap markers are excluded where applicable. Routes are restricted to regular-season WR/TE observations with recorded labels and an eligible snap. Native extraction retains exact 100 ms timestamps, with no interpolation.

The prepared route population has 42,807 player-play rows across the three seasons. Eligibility at a timestamp requires a complete observed post-snap prefix and a frame before the terminating event. Pre-snap missing observations are imputed with training-set means and accompanied by missingness indicators. The event window ends at the first recorded pass release, tackle, out-of-bounds, sack, incomplete pass, touchdown, or fumble for the tracked player.

The initial preparation stage also creates lower-rate helper arrays for cohort construction. The recognition model and integral use the subsequent native 10 Hz extraction, not those helper arrays.

## Partitions and inputs

All 2025 routes are held out from fitting. Among 2023–2024 games, `game_id % 5 == 0` defines validation; the rest are training. Validation is divided by `(game_id // 5) % 2`: zero for selection, one for calibration. These deterministic game partitions keep player-play observations from the same game together. Players may appear in both training and evaluation, so unseen-player results are reported separately.

2025 has been used for model development and must not be presented as an untouched confirmatory sample. The next confirmatory evaluation needs new games or a genuinely unused cohort.

Numerical preprocessing is fit on training rows. Own-player motion comprises relative longitudinal/lateral position, speed, acceleration, and velocity components. Coordinates are normalized for play direction and mirrored toward the field interior. Static context includes down, yards to go, clock, score, field location, alignment, formation, position, and motion-at-snap status. The neural models add observed counts, centroids, speed, velocity, and spatial spread of available offense/defense players, excluding the focal player from offense aggregates.

No target status, pass result, separation, Combine measurement, future location, player identifier, or recorded route is used as a recognition input. Future locations and route labels appear only as supervised targets. The historical-tree candidate uses smoothed player/team counts from strictly earlier training games; current-game, validation, and 2025 labels never update those counts. It is excluded from the candidate pool defining movement-only unpredictability.

## Models

The family map is explicit in [family-map.json](../results/family-map.json). It is an analyst-defined taxonomy, not a second source of ground truth. In particular, POST is in-breaking and WHEEL is vertical.

Hierarchical histogram gradient-boosted trees factor exact-route probability into family probability times route probability within family. The family model uses 130 iterations, 23 leaves, learning rate 0.08, and L2 regularization 20. Conditional models use 110 iterations and 15 leaves. Training subsamples observation times to control repeated-frame volume; prediction still runs at every native frame.

Neural candidates use 96-dimensional temporal representations, a 32-dimensional static-context embedding, and 0.15 dropout. Sequence GRU predicts routes and families. Joint route-and-path GRU adds future 1- and 2-second displacement targets with masked Smooth L1 loss. The temporal convolution model uses four causal residual layers with kernel width three and dilations 1, 2, 4, and 8; it also includes the path auxiliary objective. Route cross-entropy has weight 1, family cross-entropy 0.15, and path loss 0.3 when enabled. GRUs use up to 50 epochs, the convolution model up to 40, with seven-epoch selection patience. AdamW uses learning rate 0.001, weight decay 0.01, and gradient clipping at 2.

Candidate probabilities are temperature-calibrated on calibration games. Selection compares single models and two-model blends at weights 0.25, 0.50, and 0.75 using selection log loss. The selected blend is calibrated again on calibration games. No 2025 labels choose weights or temperature. The movement and history-allowed selection pools select the same final blend in the reference run.

Future-input perturbation checks verify temporal causality. The convolution stress check runs in double precision to prevent single-precision convolution rounding under a large perturbation from being mistaken for leakage. It reports exactly zero effect on earlier logits. Probability sums and game separation are asserted during evaluation.

The scalar path head is an auxiliary forecast, not a probability distribution over future positions. Its evaluation is reported separately in the results JSON.

## Scores, adjustment, and uncertainty

Route unpredictability is the trapezoidal integral of `1 - p_actual`, scaled by `100 / duration`. A full window spans 31 observations from 0 to 3 seconds. At least six contiguous frames are required. Scores cover the observed pre-release portion, not the route after the throw or beyond the cap. Missing frames are never bridged.

Integrated Brier loss uses the sum over all twelve labels and a factor of 50, placing it on a 0–100 range. Integrated log loss clips probabilities at `1e-8` for numerical stability. Normalized entropy is

$$
A_i=\frac{100}{T_i\log 12}\int_0^{T_i}-\sum_k p_{i,t}(k)\log p_{i,t}(k)\,dt.
$$

Entropy does not use the true label. The confidently wrong fraction counts time with an incorrect most-likely route and maximum probability at least 0.8. These diagnostics help distinguish maintained alternatives from confident model errors. A uniform twelve-route forecast gives unpredictability 91.67; it is not the midpoint of the scale.

Recognition delay is another retrospective diagnostic: the actual label must be the most likely and exceed a chosen threshold for five consecutive frames, spanning 0.4 seconds. Thresholds 0.5, 0.6, and 0.7 are reported for exact routes and families. Onset is only confirmed after the persistence window. Routes without recognition before the cutoff are censored; they are not assigned an artificial delay of three seconds.

A ridge model with penalty 20, fit only on validation routes, estimates expected unpredictability using route, position, formation, alignment, lined-up position, down, yards to go, duration, duration squared, and sideline distance. A separate ridge model adjusts Brier loss. This adjustment is descriptive; conditioning on route and duration changes the estimand and does not remove every scheme or opportunity effect.

Player estimates weight routes equally. Confidence intervals use 1,000 game-within-player bootstrap samples. Fewer than 50 routes or five games triggers a provisional flag. Empirical variance shrinkage pulls imprecise estimates toward the qualified-player mean; intervals refer to the unshrunk estimate. They do not include uncertainty from retraining the recognition model.

Repeatability divides games by game-ID parity and requires at least 25 routes in each group. It is not chronological test-retest reliability and does not prove a stable player skill. Fixed-window sensitivity compares 0–2 seconds on the same eligible routes. Alternative-model sensitivity uses the same context adjustment for each model.

## Combine measurements

The 18 prespecified features are height, weight, ten-yard split, forty-yard time, vertical jump, broad jump, three-cone time, short-shuttle time, sensor speed/acceleration 95th percentiles during the forty, and four measures each from `SLANT_ROUTE_LEFT` and `CURL_ROUTE_RIGHT`: speed 95th percentile, turn-rate 90th percentile, speed at the detected turn, and speed retention.

Forty sensor measurements aggregate attempts by median. For route drills, candidate turn frames must be inside 15–80% of traveled distance, exceed two yards per second, and have valid timing. Angular change is wrapped before differentiating. The largest eligible turn rate identifies the turn. Speed retention is mean speed over the following three frames divided by mean speed over the preceding three, with a small denominator floor. Per-player features are medians over attempts. These operational definitions require sensitivity checks before use as physical traits.

Associations are complete-case partial rank correlations. Rank residuals adjust for position, draft class, log route count, and mean observed duration. Each feature requires at least 25 players. There are 18 features × two outcomes; all 36 planned tests share a Benjamini–Hochberg correction. Untestable features would contribute p=1. Residual permutation tests use 3,000 draws; pointwise intervals use 800 player bootstrap draws. Shared scheme and nuisance-model uncertainty are not fully captured.

The separation check aggregates the provided `separation_at_pass_forward` field, requires at least 50 nonmissing observations, and uses 1,000 player bootstrap draws. The available files do not provide its full time series. The official defender-selection definition should be verified before equating the field with any newly constructed nearest-defender measure. A nearest-recorded-defender distance from incomplete cohort tracking must not be called true separation.

## Running and inspecting the analysis

`src/run.py` runs preparation, models, calibration/evaluation, integration, feature extraction, associations, and stability in order. It accepts `--data`, `--work`, and `--output`. The data directory must contain all nine CSVs; outputs are local. The pipeline was executed in Python 3.14 on Apple silicon using CPU PyTorch. Exact package versions are recorded in [environment.json](../results/environment.json); cross-version floating-point results can differ.

The notebook invokes the same pipeline and can locate data attached in a Kaggle environment. It is a reproduction entry point, not a claim of an executed or submitted Kaggle notebook. Full-training verification was performed locally; the exported source also receives syntax checks and a sensor-feature extraction smoke test.

To render figures after the pipeline, set `NFL_DATA_DIR`, `NFL_BASE_WORK`, `NFL_MODEL_WORK`, `NFL_OUTPUT_DIR`, and `NFL_FIGURE_DIR`, then run `python src/figures.py`. Default work locations match the pipeline. A separate local play visualization displays all available tracked players; that visualization and its embedded tracking data are not part of this public release.

Raw tracking, row-level route outputs, and generated play demos stay outside the public repository. Published figures and JSON files contain derived or aggregate results. Source code does not include credentials, downloaded competition data, or trained checkpoints.
