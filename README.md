# DIGER: Differentiable Semantic ID for Generative Recommendation

This is the `gradient-fix` branch of DIGER. The original paper-reproduction
path accidentally detached the recommendation-loss gradient before it reached
the RQ-VAE assignments and codebooks. This branch restores that gradient path
and is recommended for new experiments. The historical implementation remains
on [`main`](https://github.com/junchen-fu/DIGER/tree/main).

The corrected True-E2E FrQUD path uses cached hard Sinkhorn semantic IDs in the
forward pass and current soft assignments in the backward pass. Each run records
its resolved configuration and manifest, selects the checkpoint on validation,
and evaluates that checkpoint once on test.

Gumbel noise is used during training to mitigate RQ-VAE codebook collapse by
encouraging alternative code assignments. FrQUD adds this noise only to
assignments whose EMA code usage is above the configured frequency threshold.
The perturbation affects the current soft assignment in the backward path,
while the forward semantic IDs remain cached hard Sinkhorn IDs. Gumbel noise is
disabled for validation and test, which therefore use deterministic semantic
IDs.

Paper: [Differentiable Semantic ID for Generative Recommendation](https://arxiv.org/abs/2601.19711)

## Setup

```bash
conda create -n diger python=3.12.11 -y
conda activate diger
pip install -r requirements.txt
```

Released artifacts:

- [Processed data and embeddings](https://huggingface.co/datasets/junchenfu/diger-processed-data)
- [Beauty RQ-VAE checkpoint](https://huggingface.co/junchenfu/diger-rqvae-beauty)
- [Instruments RQ-VAE checkpoint](https://huggingface.co/junchenfu/diger-rqvae-instruments)
- [Yelp RQ-VAE checkpoint](https://huggingface.co/junchenfu/diger-rqvae-yelp)

Place the data under `dataset/<dataset>/` and checkpoints at
`rqvae_ckpt/<dataset>/best_collision_model.pth`, then verify them:

```bash
python scripts/check_artifacts.py
```

## Reproduction

Run one FrQUD experiment per GPU:

```bash
bash scripts/run_gradient_fix_frqud.sh beauty 0 beauty_gradient_fix
bash scripts/run_gradient_fix_frqud.sh instruments 1 instruments_gradient_fix
bash scripts/run_gradient_fix_frqud.sh yelp 2 yelp_gradient_fix
```

The runner writes the training log, manifest, validation-best checkpoint path,
test log, and test JSON under
`reproduction_logs/true_e2e_frqud/<run-label>/`. Set `DIGER_ENV_BIN`,
`DIGER_DATA_ROOT`, or `DIGER_RQVAE_ROOT` when the environment or artifacts live
outside the repository.

## Results

Metrics are `R@5 / R@10 / N@5 / N@10`. Historical FrQUD is the paper result;
gradient-fix FrQUD is the validation-selected checkpoint evaluated once on test.

| Dataset | Historical FrQUD | Gradient-fix FrQUD |
| --- | --- | --- |
| Beauty | 0.0440 / 0.0683 / 0.0294 / 0.0372 | 0.0447 / 0.0684 / 0.0293 / 0.0369 |
| Instruments | 0.0915 / 0.1138 / 0.0772 / 0.0844 | 0.0906 / 0.1126 / 0.0762 / 0.0833 |
| Yelp | 0.0266 / 0.0432 / 0.0173 / 0.0227 | 0.0259 / 0.0429 / 0.0167 / 0.0221 |

Release presets:

- Beauty: `ratio=1.5`, `tau=2.0`, `lr_id=1e-5`, `beam=20`
- Instruments: `ratio=2.0`, `tau=2.0`, `lr_id=2e-5`, `beam=20`
- Yelp: `ratio=1.1`, `tau=1.5`, `lr_id=2e-6`, `beam=80`
- All datasets: `seed=2020`, `lr_rec=1e-3`

## RQ-VAE Checkpoint Training

Use the released checkpoints for the reported experiments. To rebuild them from
semantic embeddings:

```bash
bash scripts/reproduce_rqvae.sh --embedding /path/to/Beauty.emb-llama.npy --dataset beauty
bash scripts/reproduce_rqvae.sh --emb-dir /path/to/dataset
bash scripts/reproduce_rqvae.sh --all --gpu 0,1
```

Verify the reproduced checkpoint metadata with:

```bash
python scripts/rqvae/verify_rqvae_ckpt.py
```

## Citation

```bibtex
@inproceedings{fu2026differentiable,
  author = {Fu, Junchen and Ge, Xuri and Karatzoglou, Alexandros and Arapakis, Ioannis and Verberne, Suzan and Jose, Joemon M. and Ren, Zhaochun},
  title = {Differentiable Semantic ID for Generative Recommendation},
  year = {2026},
  publisher = {Association for Computing Machinery},
  doi = {10.1145/3805712.3809641},
  booktitle = {Proceedings of the 49th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages = {369--379}
}
```
