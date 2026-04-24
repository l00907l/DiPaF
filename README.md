# DiPaF: Bridging Local Dynamics and Global Patterns for Multivariate Time Series Forecasting

**DiPaF** is a discrete pattern-aware forecasting framework that reformulates time series forecasting as a joint discrete-continuous learning process. Rather than directly regressing future values from historical inputs, DiPaF first infers discrete structural patterns as high-level priors to explicitly guide continuous value generation, effectively bridging fine-grained local dynamics with reusable global patterns.

---

## Key Features

- **Discrete-Continuous Dual Optimization**: Jointly optimizes a predictive semantic alignment loss (Cross-Entropy) and a numerical regression loss (MSE), ensuring forecasts are both structurally coherent and numerically accurate.
- **Shared Learnable Prototype Bank**: A self-supervised pre-trained prototype bank abstracts recurring temporal motifs across all channels into a unified discrete semantic space, enabling lightweight cross-variable structural alignment.
- **Pattern-guided Refinement**: Retrieved discrete structural anchors are fused with continuous latent representations to refine future predictions, alleviating over-smoothing in challenging scenarios.

---

## Getting Started

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

### 2. Download Datasets

Download all datasets from [Autoformer](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) and place all `.csv` files in the `./dataset` directory.

---

## 🏋️ Training

DiPaF adopts a **two-stage training pipeline**:

### Stage Ⅰ: Prototype Bank Pre-training

Self-supervised pre-training to initialize the encoder, decoder, and prototype bank by abstracting recurring temporal motifs into a compact discrete structural vocabulary.

All scripts are in `./scripts/VQVAE/`. Reconstruction results can be found in `./vqvae_recon_results/`, and the trained weights are saved in `/ckpt_vqvae`.

For example, to run the pre-training on the `weather` dataset, execute:
```bash
sh ./scripts/VQVAE/weather.sh
```

### Stage Ⅱ: Discrete Pattern-aware Forecasting

After obtaining the prototype bank, you can train the DiPaF forecasting model.

All scripts are in `./scripts/DiPaF/`. Forecasting results are saved in `./result.txt`, and training logs are available in `./logs/`.
To get the multivariate forecasting results for the weather dataset, run:
```bash
sh ./scripts/DiPaF/weather.sh
```

---

## 📬 Contact

If you have any questions or suggestions, feel free to reach out: **zuolingling@njust.edu.cn**

---

## 🙏 Acknowledgement

We sincerely thank the following repositories for their valuable code and datasets:

- [PatchTST](https://github.com/yuqinie98/PatchTST)
