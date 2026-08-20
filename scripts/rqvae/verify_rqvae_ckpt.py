#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch


TARGETS = {
    "beauty": {
        "epochs": 10000,
        "batch_size": 1024,
        "beta": 0.25,
        "sk_iters": 50,
        "sk_epsilons": [0.003, 0.003, 0.003],
        "num_emb_list": [256, 256, 256],
        "layers": [2048, 1024, 512],
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "e_dim": 256,
        "vq_type": "vq",
        "dist": "l2",
    },
    "instruments": {
        "epochs": 10000,
        "batch_size": 2048,
        "beta": 0.25,
        "sk_iters": 50,
        "sk_epsilons": [0.003, 0.003, 0.003],
        "num_emb_list": [256, 256, 256],
        "layers": [2048, 1024, 512],
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "e_dim": 256,
        "vq_type": "vq",
        "dist": "l2",
    },
    "yelp": {
        "epochs": 10000,
        "batch_size": 4096,
        "beta": 0.5,
        "sk_iters": 50,
        "sk_epsilons": [0.003, 0.003, 0.003],
        "num_emb_list": [256, 256, 256],
        "layers": [2048, 1024, 512],
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "e_dim": 256,
        "vq_type": "vq",
        "dist": "l2",
    },
}


def load_ckpt(path: str):
    return torch.load(path, map_location="cpu", weights_only=False)


def close_or_equal(a, b, abs_tol=1e-8):
    try:
        if isinstance(a, (list, tuple)):
            if len(a) != len(b):
                return False
            return all(abs(float(x) - float(y)) <= abs_tol for x, y in zip(a, b))
        return abs(float(a) - float(b)) <= abs_tol
    except Exception:
        return a == b


def check_dataset(ds, args_dict, ckpt, strict):
    target = TARGETS[ds]
    checks = []
    passed = True
    state = ckpt.get("state_dict", {})
    required_prefixes = ["encoder.", "rq.", "decoder."]
    state_ok = all(any(k.startswith(prefix) for k in state.keys()) for prefix in required_prefixes)
    if not state_ok:
        passed = False

    for key, expected in target.items():
        actual = args_dict.get(key)
        if key in {"sk_epsilons", "num_emb_list", "layers"}:
            actual = list(actual) if actual is not None else None
        ok = close_or_equal(actual, expected) if actual is not None else False
        checks.append((key, actual, expected, ok))
        if not ok:
            passed = False

    checkpoint_epoch = ckpt.get("epoch")
    status = "PASS" if passed else "WARN"
    print(
        f"{status} {ds:10s} | ckpt_epoch={checkpoint_epoch} | "
        f"best_collision={ckpt.get('best_collision_rate', float('nan')):.8f}"
    )
    print(f"  - state_prefixes: {state_ok}")
    for key, actual, expected, ok in checks:
        print(f"  - {key}: {actual} | target={expected} | {'OK' if ok else 'BAD'}")

    if strict and not passed:
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_root", default="./rqvae_ckpt")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(TARGETS),
        default=sorted(TARGETS),
    )
    args = parser.parse_args()

    root = Path(args.ckpt_root)
    all_ok = True

    for ds in args.datasets:
        ckpt_path = root / ds / "best_collision_model.pth"
        if not ckpt_path.exists():
            print(f"MISSING {ds}: {ckpt_path}")
            all_ok = False
            continue

        try:
            ckpt = load_ckpt(str(ckpt_path))
        except Exception as exc:
            print(f"FAILED_LOAD {ds}: {exc}")
            all_ok = False
            continue

        args_dict = ckpt.get("args")
        if hasattr(args_dict, "__dict__"):
            args_dict = dict(args_dict.__dict__)

        if not isinstance(args_dict, dict):
            print(f"UNSUPPORTED_ARGS_FORMAT {ds}: type={type(args_dict)}")
            all_ok = False
            continue

        ok = check_dataset(ds, args_dict, ckpt, args.strict)
        all_ok = all_ok and ok

    print("\nSummary:")
    if all_ok:
        print("All datasets pass target config validation.")
        return 0
    print("Validation failed for one or more datasets.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
