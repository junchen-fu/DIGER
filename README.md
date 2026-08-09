# DIGER: Differentiable Semantic ID for Generative Recommendation

**SIGIR 2026 full paper.** DIGER learns differentiable semantic IDs for generative recommendation, allowing item identifiers to be optimized jointly with the recommendation model.

![SID](https://img.shields.io/badge/Task-SID-red)
![Generative recommendation](https://img.shields.io/badge/Task-Generative--recommendation-red)
<a href="https://arxiv.org/abs/2601.19711" alt="arXiv"><img src="https://img.shields.io/badge/arXiv-2601.19711-FAA41F.svg?style=flat" /></a>
<a href="https://mp.weixin.qq.com/s/Cs2kwRR0U94GyT5h7hkldg" alt="Chinese blog"><img src="https://img.shields.io/badge/blog-新智元-orange.svg?style=flat" /></a>
<a href="https://github.com/jhljx/RecSys-Industrial-Book/blob/main/ch09%20%E7%94%9F%E6%88%90%E5%BC%8F%E6%8E%A8%E8%8D%90%E6%A8%A1%E5%9D%97/%E7%AC%AC%E4%B9%9D%E7%AB%A0-%E7%94%9F%E6%88%90%E5%BC%8F%E6%8E%A8%E8%8D%90%E6%A8%A1%E5%9D%97.pdf" alt="Book"><img src="https://img.shields.io/badge/Book-%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F·%E5%B7%A5%E4%B8%9A%E6%9E%B6%E6%9E%84-2ea44f.svg?style=flat" /></a>

## News

- **[2026-08]** Honored to be included in the generative recommender chapter of [RecSys-Industrial-Book](https://github.com/jhljx/RecSys-Industrial-Book/blob/main/ch09%20%E7%94%9F%E6%88%90%E5%BC%8F%E6%8E%A8%E8%8D%90%E6%A8%A1%E5%9D%97/%E7%AC%AC%E4%B9%9D%E7%AB%A0-%E7%94%9F%E6%88%90%E5%BC%8F%E6%8E%A8%E8%8D%90%E6%A8%A1%E5%9D%97.pdf) / 《推荐系统：工业架构与核心算法》.

<p align="center">
  <img src="assets/figure1.png" alt="Conventional versus differentiable semantic IDs in generative recommendation" width="80%" />
</p>

*Conventional pipelines freeze RQ-VAE semantic IDs, while DIGER makes semantic IDs differentiable and optimizes them together with the recommendation model.*

<p align="center">
  <img src="assets/figure3.png" alt="DIGER framework compared with straight-through estimation" width="80%" />
</p>

*DIGER uses stochastic exploration with Gumbel noise and uncertainty decay to support a stable exploration-to-exploitation transition.*

## Overview

This repository contains the code, processed data, semantic embeddings, and RQ-VAE checkpoints needed to reproduce the released DIGER results for:

- **FrqUD**: frequency-based uncertainty decay.
- **SDUD**: standard-deviation uncertainty decay.
- **SDUD+FrqUD**: the combined setting.

The released scripts cover all three datasets used in the table: Beauty, Instruments, and Yelp.

## Release Artifacts

- Code: this repository.
- Processed data and embeddings: [junchenfu/diger-processed-data](https://huggingface.co/datasets/junchenfu/diger-processed-data).
- RQ-VAE checkpoints: [Beauty](https://huggingface.co/junchenfu/diger-rqvae-beauty), [Instruments](https://huggingface.co/junchenfu/diger-rqvae-instruments), and [Yelp](https://huggingface.co/junchenfu/diger-rqvae-yelp).

## Repository Structure

```text
DIGER/
├── accelerate_config.yaml
├── accelerate_config_multi_gpu.yaml
├── main.py
├── model.py
├── trainer.py
├── vq.py
├── data.py
├── config/
│   ├── beauty_jo.yaml
│   ├── instruments_jo.yaml
│   └── yelp_jo.yaml
├── dataset/
│   ├── beauty/
│   ├── instruments/
│   └── yelp/
├── rqvae_ckpt/
│   ├── beauty/best_collision_model.pth
│   ├── beauty/best_collision_model_two_gpu_preview.pth
│   ├── instruments/best_collision_model.pth
│   └── yelp/best_collision_model.pth
├── scripts/
│   ├── check_artifacts.py
│   ├── run_experiment.sh
│   ├── run_experiment_two_gpus.sh
│   ├── run_table_two_gpus.sh
│   ├── run_rqvae_pretrain.sh
│   ├── rqvae/
│   │   ├── main.py
│   │   ├── datasets.py
│   │   ├── trainer.py
│   │   ├── utils.py
│   │   ├── verify_rqvae_ckpt.py
│   │   └── models/
│   └── verify_results.py
├── run_FrqUD.sh
├── run_SDUD.sh
├── run_SDUD_FrqUD.sh
├── run_rqvae_beauty.sh
├── run_rqvae_instruments.sh
├── run_rqvae_yelp.sh
├── run_rqvae_all.sh
└── run_reproduce_table.sh
```

Large local artifacts, including checkpoints, embeddings, and JSONL splits, are tracked with Git LFS. They are also available from the Hugging Face links above.

## Requirements

```bash
conda create -n diger python=3.12.11 -y
conda activate diger
pip install -r requirements.txt
```

Reference environment used for the released paper logs:

- Python 3.12.11
- PyTorch 2.5.1
- Transformers 4.57.1
- Accelerate 1.10.1
- NumPy 2.3.1

Using newer major versions can change initialization and dropout RNG streams. One quick environment sanity check is the first Yelp SDUD+FrqUD training line:

```text
[Simple Uncertainty] sigma=2.0000, Loss=5.4814
```

If this line is closer to `Loss=5.5120`, the code and artifacts are likely correct, but the active Python environment may differ from the reference environment.

After cloning the repository, pull the LFS files:

```bash
git lfs install
git lfs pull
```

Then verify the released artifacts:

```bash
python scripts/check_artifacts.py
```

## Data

Each dataset directory contains JSONL interaction splits, an item-id map, and the semantic embedding matrix used by the released configs:

```text
dataset/<dataset>/
├── <dataset>.train.jsonl
├── <dataset>.valid.jsonl
├── <dataset>.test.jsonl
├── <dataset>.emb_map.json
└── <Dataset>.emb-llama.npy
```

The loader expects each JSONL row to contain `inter_history` and `target_id`.

## Reproduction

Run one experiment:

```bash
bash scripts/run_experiment.sh beauty frqud
bash scripts/run_experiment.sh instruments sdud
bash scripts/run_experiment.sh yelp both
```

### Two-GPU Low-Memory (Preview)

> [!WARNING]
> **Not an exact paper reproduction yet.** The verified Beauty SDUD run reached **R@5 = 0.043375**, versus **0.044180** in the paper (**-1.82%**). Full parity with the paper result remains TODO.

For two 24 GB GPUs:

```bash
GPU=0,1 bash scripts/run_experiment_two_gpus.sh beauty sdud
```

This runs DDP with `32 per device x 2 GPUs x 4 accumulation = 256`, gradient checkpointing, and evaluation batch size 4. It uses a separate preview RQ-VAE checkpoint; the standard single-GPU path is unchanged.

If you keep the paper environment outside your current shell, point the script at its `bin` directory:

```bash
DIGER_ENV_BIN=/path/to/env/bin bash scripts/run_experiment.sh yelp both
```

The same variable works for the full-table launcher:

```bash
DIGER_ENV_BIN=/path/to/env/bin bash run_reproduce_table.sh
```

Convenience wrappers default to Beauty and accept the dataset as the first argument:

```bash
bash run_FrqUD.sh beauty
bash run_SDUD.sh instruments
bash run_SDUD_FrqUD.sh yelp
```

Run the full paper table on at most two single-GPU processes. The worker script uses a shared task queue, so whichever GPU finishes first takes the next experiment:

```bash
bash run_reproduce_table.sh
```

By default this uses GPU `0` and GPU `1`. To choose another pair:

```bash
GPU_LIST="2 3" bash run_reproduce_table.sh
```

By default, `run_reproduce_table.sh` starts a fresh queue by resetting `reproduction_logs/table_queue.state`. To resume an interrupted queue, set:

```bash
RESUME_QUEUE=1 bash run_reproduce_table.sh
```

Training logs are written to `logs/<dataset>/`; stdout mirrors are written to `reproduction_logs/`. Model checkpoints are written to `myckpt/<dataset>/`.

## RQ-VAE Checkpoint Training

This repository also includes the RQ-VAE pretraining implementation from `scripts/rqvae/`.  
You can train the released RQ-VAE checkpoints directly from embeddings.

Default hyper-parameters:

- `lr=1e-3`, `weight_decay=1e-4`
- `epochs=10000`
- `batch_size`: beauty 1024, instruments 2048, yelp 4096
- `num_emb_list=[256,256,256]`
- `layers=[2048,1024,512]`
- `e_dim=256`, `beta=0.25` (beauty/instruments), `beta=0.5` (yelp)
- `sk_epsilons=[0.003,0.003,0.003]`, `sk_iters=50`
- `vq_type=vq`, `loss_type=mse`, `dist=l2`, `kmeans_init=True`

Run one dataset:

```bash
bash scripts/run_rqvae_beauty.sh
bash scripts/run_rqvae_instruments.sh
bash scripts/run_rqvae_yelp.sh
```

Run all three sequentially:

```bash
bash scripts/run_rqvae_all.sh
```

Or if you already have an embedding file for one dataset, jump directly:

```bash
bash scripts/run_rqvae_from_embedding.sh --embedding /path/to/Beauty.emb-llama.npy
bash scripts/run_rqvae_from_embedding.sh --embedding /path/to/custom_embedding.npy --dataset beauty
```

LLaMA embeddings referenced here should be generated following the instructions in [honghuibao2000/letter](https://github.com/honghuibao2000/letter).

You can also use a unified RQ-VAE training wrapper (recommended):

```bash
bash scripts/reproduce_rqvae.sh --embedding /path/to/Beauty.emb-llama.npy --dataset beauty
bash scripts/reproduce_rqvae.sh --embedding /path/to/custom_embedding.npy --dataset yelp --gpu 0,1
bash scripts/reproduce_rqvae.sh --emb-dir /path/to/dataset
bash scripts/reproduce_rqvae.sh --all --gpu 0,1
```

GPU control (important):

```bash
RQVAE_GPU="0" bash scripts/run_rqvae_from_embedding.sh --embedding ...
RQVAE_GPU="0,1" bash scripts/run_rqvae_from_all_embeddings.sh
```

The runner accepts one or two GPU ids and will stop with an error if more than two are requested.

Notes:

- RQ-VAE implementation files in this repository are kept in `scripts/rqvae/` and include `main.py`, `trainer.py`, `datasets.py`, `utils.py`, and `models/{rq.py,layers.py,rqvae.py,vq.py}`.
- Use this section as a public training workflow. If you need to compare against your own baseline checkpoints, set `--baseline_root` and `--ckpt_root` to your local paths.

After training, run optional verification (for configuration, collision, and epoch consistency):

```bash
python3 scripts/rqvae/compare_rqvae_ckpt.py --ckpt_root ./rqvae_ckpt --baseline_root /path/to/baseline_rqvae_ckpt --expect_hash --strict
```

Run all three default embeddings from the repo in one go:

```bash
bash scripts/run_rqvae_from_all_embeddings.sh
``` 

All runners copy the newest `best_collision_model.pth` to:

```text
rqvae_ckpt/<dataset>/best_collision_model.pth
```

Checkpoint lineage:

- `scripts/rqvae/` in this repo is the core RQ-VAE pretraining implementation used for this release.
- The code is aligned with the corresponding RQ-VAE implementation in the original project.

If you want to train under a custom output directory from a custom embedding, set `RQVAE_CKPT_ROOT` explicitly:

```bash
RQVAE_CKPT_ROOT=/your/ckpt/root RQVAE_EPOCHS=20000 \
  bash scripts/run_rqvae_from_embedding.sh --embedding /path/to/your_llm_embedding.npy --dataset beauty
```

The script copies the best checkpoint to:

```text
${RQVAE_CKPT_ROOT}/beauty/best_collision_model.pth
```

Check that your RQ-VAE checkpoint metadata matches the released setup:

```bash
python scripts/rqvae/verify_rqvae_ckpt.py
```

## Paper Data

The paper reports these metrics (R@5/R@10/N@5/N@10):

| Dataset | Variant | R@5 | R@10 | N@5 | N@10 |
| --- | --- | ---: | ---: | ---: | ---: |
| Beauty | DIGER (FrqUD) | 0.0440 | 0.0683 | 0.0294 | 0.0372 |
| Beauty | DIGER (SDUD) | 0.0442 | 0.0657 | 0.0292 | 0.0361 |
| Beauty | DIGER (SDUD+FrqUD) | 0.0439 | 0.0696 | 0.0293 | 0.0376 |
| Instruments | DIGER (FrqUD) | 0.0915 | 0.1138 | 0.0772 | 0.0844 |
| Instruments | DIGER (SDUD) | 0.0905 | 0.1124 | 0.0753 | 0.0823 |
| Instruments | DIGER (SDUD+FrqUD) | 0.0907 | 0.1127 | 0.0758 | 0.0829 |
| Yelp | DIGER (FrqUD) | 0.0266 | 0.0432 | 0.0173 | 0.0227 |
| Yelp | DIGER (SDUD) | 0.0267 | 0.0439 | 0.0171 | 0.0227 |
| Yelp | DIGER (SDUD+FrqUD) | 0.0273 | 0.0437 | 0.0175 | 0.0227 |

After training, compare the newest matching logs with the paper targets:

```bash
python scripts/verify_results.py
```

The verifier scans `logs/*/*.log` and `reproduction_logs/*` (including `.driver.log`) from the current run. It uses a 1% relative tolerance with a small absolute floor for very small metrics.

If you only want to check the packaged reference logs without rerunning experiments, run:

```bash
VERIFY_WITH_BACKUP=1 python scripts/verify_results.py
```

## Citation

```bibtex
@inproceedings{fu2026differentiable,
author = {Fu, Junchen and Ge, Xuri and Karatzoglou, Alexandros and Arapakis, Ioannis and Verberne, Suzan and Jose, Joemon M. and Ren, Zhaochun},
title = {Differentiable Semantic ID for Generative Recommendation},
year = {2026},
isbn = {9798400725999},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3805712.3809641},
doi = {10.1145/3805712.3809641},
booktitle = {Proceedings of the 49th International ACM SIGIR Conference on Research and Development in Information Retrieval},
pages = {369–379},
numpages = {11},
series = {SIGIR '26}
}
```
