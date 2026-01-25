# PALEOv2
PALEO: Physics Aware Lightkurve Exoplanet Observer is a Temporal Convolutional Network with an integrated physics aware head to classify and detect Earth-Like Exoplanet's whose transits are submerged in low Signal-To-Noise Ratios. 
PALEO

Physics-Aware Lightcurve Event Observer

PALEO is a physics-aware deep learning pipeline for detecting exoplanet transits in stellar light curves. It combines a Temporal Convolutional Neural Network (TCN) with physical transit modeling to identify, validate, and reject candidate transit signals in noisy Kepler data—particularly in the presence of stellar activity such as flares and rotational modulation.

Rather than treating transit detection as a purely black-box classification task, PALEO explicitly integrates astrophysical constraints into both labeling and post-inference analysis.

Key Features

Multi-label event detection
Simultaneously detects:

Planetary transit events

Stellar flares

Background variability (rotational modulation + noise)

Physics-aware validation
Candidate transits predicted by the neural network are validated using analytic transit models generated with the BATMAN package.

Time-resolved probability outputs
The model produces per-cadence probability curves, allowing direct alignment between predictions and physical light-curve features.

Windowed but temporally consistent analysis
Light curves are segmented into overlapping windows for training, while prediction consistency is analyzed across overlapping regions in the original, unsegmented time series.

Methodology Overview

Data Source
Public Kepler long-cadence light curves accessed via the MAST archive using the lightkurve Python package.

Preprocessing & Windowing

Light curves are normalized and segmented into fixed-length overlapping windows.

Gaps and low-coverage regions are masked to preserve temporal integrity.

Labeling Strategy

Transit labels are generated using known planetary ephemerides (period, epoch, duration).

Flare labels are identified using robust, local outlier detection methods.

Labels are assigned per cadence, enabling fine-grained temporal supervision.

Model Architecture

A Temporal Convolutional Network (TCN) with dilated causal convolutions is used to capture long-range temporal dependencies.

The model outputs independent probability channels for each event type.

Physics-Aware Analysis

High-confidence transit regions are extracted from model outputs.

Synthetic transit curves are generated using BATMAN and overlaid on candidate regions.

Physical consistency is evaluated using a noise-normalized RMSE metric to accept or reject candidates.

Why PALEO?

Traditional machine-learning approaches to transit detection often:

Ignore physical interpretability

Collapse temporal structure into a single classification score

Struggle with stellar activity and false positives

PALEO addresses these limitations by:

Preserving time-resolved predictions

Separating astrophysical events explicitly

Enforcing physical plausibility through post-model validation

This makes PALEO particularly well-suited for detecting small, Earth-like planets around active stars.

Repository Structure
