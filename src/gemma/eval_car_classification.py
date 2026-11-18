"""
eval_car_classification.py
Zero-shot evaluation of Gemma on the same Stanford Cars test split used for the
ResNet-50 baseline (same stratified split, random_state=42, bbox-cropped images).

Two modes:
  closed - model must pick one of the 196 Stanford Cars class names (apples-to-apples
           with the ResNet's 84.37% test accuracy)
  open   - model freely names make/model/year (closer to real-world use); scored on
           make accuracy, make+model accuracy, and year within +/-1

Results are appended to results/gemma_cars_<mode>_<model>.jsonl, so an interrupted
run resumes where it left off.

Usage (from repo root):
    python -m src.gemma.eval_car_classification --mode closed --limit 200
    python -m src.gemma.eval_car_classification --mode open --model gemma4:12b
"""

import argparse
import json
import os
import re

from PIL import Image
from tqdm import tqdm

from src.carclassification.data_loader import get_dataloaders
from src.gemma.client import DEFAULT_MODEL, ask_json, check_server

DATA_ROOT = "data/carclassificationdataset1"
IMG_DIR = os.path.join(DATA_ROOT, "cars_train", "cars_train")


def get_test_df():
    """The exact test split the ResNet was evaluated on."""
    _, _, test_loader, class_names = get_dataloaders(DATA_ROOT)
    df = test_loader.dataset.df.copy()
    df["filename"] = df["filename"].astype(str)
    df["class_name"] = df["class_name"].astype(str)
    return df, [str(c) for c in class_names]


def closed_prompt_and_schema(class_names):
    prompt = ("Identify this car. Choose the single best match from this list of "
              "'Make Model BodyStyle Year' names:\n" + "\n".join(class_names))
    schema = {
        "type": "object",
        "properties": {"class_name": {"type": "string", "enum": class_names}},
        "required": ["class_name"],
    }
    return prompt, schema


OPEN_PROMPT = ("Identify this car. Give the manufacturer (make), the model name "
               "(including trim or body style if visible), and your best guess of the model year.")
OPEN_SCHEMA = {
    "type": "object",
    "properties": {
        "make": {"type": "string"},
        "model": {"type": "string"},
        "year": {"type": "integer"},
    },
    "required": ["make", "model", "year"],
}


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _tokens(s):
    return [t for t in re.split(r"[^a-z0-9]+", s.lower()) if t]


def score_open(pred, true_name):
    """Make = class name starts with predicted make; model = every predicted model
    token appears in the class name; year = within +/-1 of the true year."""
    true_year = int(true_name.split()[-1])
    make_ok = bool(_norm(pred["make"])) and _norm(true_name).startswith(_norm(pred["make"]))
    true_tokens = set(_tokens(true_name))
    model_tokens = _tokens(pred["model"])
    model_ok = make_ok and bool(model_tokens) and all(t in true_tokens for t in model_tokens)
    return {
        "make_correct": make_ok,
        "make_model_correct": model_ok,
        "year_within_1": abs(pred["year"] - true_year) <= 1,
    }


def load_done(path):
    done = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                done[rec["filename"]] = rec
    return done


def summarize(records, mode):
    n = len(records)
    if n == 0:
        print("No results yet.")
        return
    errors = sum(1 for r in records if "error" in r)
    ok = [r for r in records if "error" not in r]
    print(f"\n{n} images evaluated ({errors} errors)")
    if mode == "closed":
        acc = sum(r["correct"] for r in ok) / n
        print(f"Exact class accuracy: {acc:.2%}   (ResNet-50 baseline: 84.37%)")
    else:
        for key in ("make_correct", "make_model_correct", "year_within_1"):
            print(f"{key:>20}: {sum(r[key] for r in ok) / n:.2%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["closed", "open"], default="closed")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=None, help="evaluate a random subset of N test images")
    parser.add_argument("--no-crop", action="store_true", help="use full image instead of bbox crop")
    args = parser.parse_args()

    check_server(args.model)

    df, class_names = get_test_df()
    if args.limit:
        df = df.sample(n=min(args.limit, len(df)), random_state=0)

    os.makedirs("results", exist_ok=True)
    out_path = f"results/gemma_cars_{args.mode}_{args.model.replace(':', '-')}.jsonl"
    done = load_done(out_path)

    if args.mode == "closed":
        prompt, schema = closed_prompt_and_schema(class_names)
    else:
        prompt, schema = OPEN_PROMPT, OPEN_SCHEMA

    with open(out_path, "a") as out:
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{args.model} [{args.mode}]"):
            if row["filename"] in done:
                continue
            img = Image.open(os.path.join(IMG_DIR, row["filename"]))
            if not args.no_crop:
                img = img.crop((row["x1"], row["y1"], row["x2"], row["y2"]))

            rec = {"filename": row["filename"], "true": row["class_name"]}
            try:
                pred = ask_json(prompt, img, schema, model=args.model)
                rec["pred"] = pred
                if args.mode == "closed":
                    rec["correct"] = pred["class_name"] == row["class_name"]
                else:
                    rec.update(score_open(pred, row["class_name"]))
            except Exception as e:  # keep going; one bad response shouldn't kill a long run
                rec["error"] = repr(e)

            out.write(json.dumps(rec) + "\n")
            out.flush()
            done[row["filename"]] = rec

    summarize([done[f] for f in df["filename"] if f in done], args.mode)


if __name__ == "__main__":
    main()
