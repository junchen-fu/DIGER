#!/usr/bin/env python3
"""Strictly load a DIGER checkpoint and evaluate one requested split."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import T5Config, T5ForConditionalGeneration


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data import Collator, SequentialSplitDataset, load_split_data  # noqa: E402
from metrics import ndcg_at_k  # noqa: E402
from model import Model  # noqa: E402
from utils import load_torch_checkpoint  # noqa: E402
from vq import RQVAE  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Strictly load a DIGER checkpoint bundle and evaluate it on validation or test. "
            "The .pt.rqvae and .code.json files must be next to the main .pt file."
        )
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "config" / "beauty_gradient_fix.yaml"
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--ignore-manifest-config",
        action="store_true",
        help="Do not merge the run manifest's resolved_config into the YAML config.",
    )
    parser.add_argument("--split", choices=("valid", "test"), default="test")
    parser.add_argument(
        "--data-path",
        "--data_path",
        type=Path,
        default=None,
        help="Optional read-only dataset root override.",
    )
    parser.add_argument("--eval-batch-size", "--eval_batch_size", type=int, default=None)
    parser.add_argument("--num-workers", "--num_workers", type=int, default=None)
    parser.add_argument("--num-beams", "--num_beams", type=int, default=None)
    parser.add_argument(
        "--max-eval-batches", "--max_eval_batches", type=int, default=0
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def resolve_path(path):
    path = path.expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def checkpoint_files(checkpoint):
    if checkpoint.suffix != ".pt":
        raise ValueError(f"Main checkpoint must end in .pt: {checkpoint}")

    files = {
        "model": checkpoint,
        "rqvae": Path(f"{checkpoint}.rqvae"),
        "codes": checkpoint.with_suffix(".code.json"),
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing checkpoint bundle file(s): " + ", ".join(missing))
    return files


def checkpoint_state(path):
    checkpoint = load_torch_checkpoint(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected a state dict in {path}")
    return checkpoint


def strict_load(model, state, path):
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise RuntimeError(f"Checkpoint is incompatible with the current code: {path}\n{error}") from error


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_models(config, num_items, model_state, rqvae_state, files):
    t5_config = T5Config(
        num_layers=config["encoder_layers"],
        num_decoder_layers=config["decoder_layers"],
        d_model=config["d_model"],
        d_ff=config["d_ff"],
        num_heads=config["num_heads"],
        d_kv=config["d_kv"],
        dropout_rate=config["dropout_rate"],
        activation_function=config["activation_function"],
        vocab_size=1,
        pad_token_id=0,
        eos_token_id=300,
        decoder_start_token_id=0,
        feed_forward_proj=config["feed_forward_proj"],
        n_positions=config["max_length"],
    )
    recommender = Model(
        config=config,
        model=T5ForConditionalGeneration(config=t5_config),
        n_items=num_items,
        code_length=config["code_length"],
        code_number=config["code_num"],
    )
    strict_load(recommender, model_state, files["model"])

    rqvae_config = dict(config)
    rqvae_config["use_learnable_sigma_gumbel"] = any(
        "auto_sigma_module" in key for key in rqvae_state
    )
    rqvae = RQVAE(config=rqvae_config, in_dim=recommender.semantic_hidden_size)
    strict_load(rqvae, rqvae_state, files["rqvae"])
    return recommender


def validate_codes(code, num_items, code_length, path):
    if len(code) != num_items:
        raise ValueError(f"{path} has {len(code)} rows; expected {num_items}")
    invalid_rows = [index for index, row in enumerate(code) if len(row) != code_length]
    if invalid_rows:
        raise ValueError(
            f"{path} contains rows with the wrong code length; first row: {invalid_rows[0]}"
        )


def batch_metrics(outputs, labels):
    totals = {
        "recall@1": 0.0,
        "recall@5": 0.0,
        "recall@10": 0.0,
        "ndcg@5": 0.0,
        "ndcg@10": 0.0,
    }
    for output, label in zip(outputs, labels):
        matches = torch.all(output == label.unsqueeze(0), dim=1).cpu().numpy()
        totals["recall@1"] += matches[:1].sum()
        totals["recall@5"] += matches[:5].sum()
        totals["recall@10"] += matches[:10].sum()
        totals["ndcg@5"] += ndcg_at_k(matches, 5)
        totals["ndcg@10"] += ndcg_at_k(matches, 10)
    return totals


@torch.inference_mode()
def evaluate(model, data_loader, code, device, max_eval_batches=0):
    model.to(device)
    model.device = device
    model.eval()
    item_code = torch.tensor(code, dtype=torch.long, device=device)
    totals = {metric: 0.0 for metric in (
        "recall@1", "recall@5", "recall@10", "ndcg@5", "ndcg@10"
    )}
    sample_count = 0

    for batch_index, batch in enumerate(tqdm(data_loader, desc="Evaluate", ncols=100)):
        if max_eval_batches > 0 and batch_index >= max_eval_batches:
            break
        input_ids = batch["input_ids"].to(device)
        labels = batch["targets"].to(device)
        batch_size = input_ids.size(0)

        input_ids = item_code[input_ids].reshape(batch_size, -1)
        labels = item_code[labels].reshape(batch_size, -1)
        attention_mask = input_ids.ne(-1)
        predictions = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            n_return_sequences=10,
        )

        current = batch_metrics(predictions, labels)
        for metric, value in current.items():
            totals[metric] += value
        sample_count += batch_size

    return {metric: round(value / sample_count, 6) for metric, value in totals.items()}


def main():
    args = parse_args()
    config_path = resolve_path(args.config)
    checkpoint = resolve_path(args.checkpoint)
    files = checkpoint_files(checkpoint)

    with config_path.open("r") as file_obj:
        config = yaml.safe_load(file_obj)
    manifest_path = checkpoint.parent / "manifest.json"
    manifest_config_used = False
    if manifest_path.is_file() and not args.ignore_manifest_config:
        with manifest_path.open("r") as file_obj:
            manifest = json.load(file_obj)
        resolved_config = manifest.get("resolved_config")
        if isinstance(resolved_config, dict):
            config.update(resolved_config)
            manifest_config_used = True
    if args.eval_batch_size is not None:
        config["eval_batch_size"] = args.eval_batch_size
    if args.num_workers is not None:
        config["num_workers"] = args.num_workers
    if args.num_beams is not None:
        config["num_beams"] = args.num_beams
    data_path = args.data_path if args.data_path is not None else Path(config["data_path"])
    config["data_path"] = str(resolve_path(data_path))

    _, num_items, _, valid, test = load_split_data(config)
    model_state = checkpoint_state(files["model"])
    rqvae_state = checkpoint_state(files["rqvae"])
    model = build_models(config, num_items, model_state, rqvae_state, files)

    with files["codes"].open("r") as file_obj:
        code = json.load(file_obj)
    validate_codes(code, num_items, config["code_length"], files["codes"])

    selected_split = valid if args.split == "valid" else test
    evaluation_dataset = SequentialSplitDataset(config, num_items, selected_split)
    collator = Collator(eos_token_id=-1, pad_token_id=0, max_length=config["max_length"])
    data_loader = DataLoader(
        evaluation_dataset,
        batch_size=config["eval_batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=args.device.startswith("cuda"),
        collate_fn=collator,
    )

    result = {
        "checkpoint": str(checkpoint),
        "manifest": str(manifest_path) if manifest_path.is_file() else None,
        "manifest_config_used": manifest_config_used,
        "sha256": {name: sha256(path) for name, path in files.items()},
        "dataset": config["dataset"],
        "split": args.split,
        "num_beams": config["num_beams"],
        "eval_batch_size": config["eval_batch_size"],
        "samples": len(evaluation_dataset),
        "evaluated_samples": min(
            len(evaluation_dataset),
            args.max_eval_batches * config["eval_batch_size"]
        ) if args.max_eval_batches > 0 else len(evaluation_dataset),
        "metrics": evaluate(
            model,
            data_loader,
            code,
            torch.device(args.device),
            max_eval_batches=args.max_eval_batches,
        ),
    }
    output = json.dumps(result, indent=2, sort_keys=True)
    print(output)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n")


if __name__ == "__main__":
    main()
