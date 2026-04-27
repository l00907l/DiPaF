# LSPP: Injecting Local Structural Patterns into Time Series Forecasting via Adaptive Prototype Pool

We propose the Local Structural Prototype Pool (LSPP), which injects local structural patterns into forecasting via an adaptive prototype pool, reformulating prediction as a discrete-continuous representation learning process.

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

LSPP adopts a **two-stage training pipeline**:

### Stage Ⅰ: Prototype Pool Pre-training

Self-supervised pre-training to initialize the encoder, decoder, and prototype pool by capturing local structural patterns into a prototype pool.

All scripts are in `./scripts/VQVAE/`. Reconstruction results can be found in `./vqvae_recon_results/`, and the trained weights are saved in `/ckpt_vqvae`.

For example, to run the pre-training on the `weather` dataset, execute:
```bash
sh ./scripts/VQVAE/weather.sh
```

### Stage Ⅱ: Pattern-aware Forecasting

After obtaining the prototype bank, you can train the LSPP forecasting model.

All scripts are in `./scripts/LSPP/`. Forecasting results are saved in `./result.txt`, and training logs are available in `./logs/`.
To get the multivariate forecasting results for the weather dataset, run:
```bash
sh ./scripts/LSPP/weather.sh
```

---

## 📬 Contact

If you have any questions or suggestions, feel free to reach out: **zuolingling@njust.edu.cn**

---

## 🙏 Acknowledgement

We sincerely thank the following repositories for their valuable code and datasets:

- [PatchTST](https://github.com/yuqinie98/PatchTST)
