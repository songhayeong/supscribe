# From Structure to Strategy: A Transformer-VAE Approach to Modeling IVF Decision Latents

This repository accompanies our research on modeling strategic treatment behavior in in-vitro fertilization (IVF) through deep representation learning.

## Overview

IVF treatment decisions are often driven not only by clinical indications but also by socioeconomic factors, behavioral strategies, and latent preferences. However, prior research has largely focused on predicting IVF success using static clinical features, overlooking the structural complexity of patient choices.

In this paper, we propose a Transformer-VAE hybrid architecture to model such decision structures in a learnable latent space, enabling interpretable simulations and counterfactual analyses.

## Research Objective

We aim to:

- Encode the interdependent structure of IVF treatment choices using a TabTransformer model,
- Compress the hidden representation into a disentangled latent vector via a Variational Autoencoder (VAE),
- Interpret and manipulate the latent space to simulate how strategic variations (e.g., elective ICSI) influence IVF outcomes,
- Explore how patient strategies reflect underlying socioeconomic behavior patterns.

## Model Architecture

- Input (Categorical + Numerical)

- TabTransformer Encoder

- VAE Encoder

- Latent Vector z ~ q(z|x)

┌────────────┐ ┌────────────┐
│ Decoder │ │ Classifier │
│ (Reconstruct) (Predict IVF Success)
└────────────┘ └────────────┘


- **TabTransformer**: Captures non-linear interactions across heterogeneous treatment and patient features using self-attention.
- **VAE**: Enables latent space compression and generative capabilities for simulation.
- **Classifier**: Connects latent structure with IVF success to assess its predictive relevance.
- **Simulation**: Adjust latent vectors to emulate strategic treatment shifts and evaluate causal impact.

## Experiments

- **Latent Disentanglement**: Assess separation in latent space between treatment strategies (e.g., Elective vs. Non-elective ICSI).
- **Counterfactual Simulation**: Modify latent dimensions and observe impact on IVF success probabilities.
- **Comparative Analysis**: Benchmark against XGBoost and MLP baselines using SHAP, reconstruction accuracy, and interpretability metrics.

## Key Contributions

- Proposes a novel Transformer-VAE pipeline for modeling strategic decision structures in medical treatment datasets.
- Bridges structural behavioral modeling with generative latent space learning.
- Demonstrates that socioeconomic treatment behaviors can be encoded and manipulated through AI for simulation and policy insights.

## Future Directions

- Integrate reinforcement learning for strategy recommendation.
- Extend to other high-cost medical treatments beyond IVF.
- Combine with causal inference frameworks to refine intervention analysis.

---

For implementation details, see [`/models/tabvae.py`](models/tabvae.py) and data preprocessing steps in [`/data`](data/).


