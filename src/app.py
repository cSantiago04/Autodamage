"""
app.py
Local demo page for the Autodamage pipeline: upload a photo of a damaged car
(optionally a VIN) and see the detected damage, the parts it's on, and the likely
repair for each, drawn on the photo.

Run (from repo root):
    python -m src.app
    python -m src.app --part-checkpoint models/part_detector/maskrcnn_hitl_finetuned_best.pth
Then open the printed local URL (http://127.0.0.1:7860). It only runs on this machine.
"""

import argparse
import colorsys
import os

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src import pipeline

MAX_SIDE = 1600  # phone photos are ~4000 px; downscale for speed and GPU memory
EXAMPLES_DIR = "data/Test Images"

ACTION_COLORS = {  # damage outline color by repair action
    "refinish only": (46, 204, 113), "paintless dent repair": (46, 204, 113),
    "repair": (241, 196, 15), "repair and refinish": (243, 156, 18), "replace": (231, 76, 60),
}


def _part_color(name):
    """Stable, distinct-ish color per part name."""
    h = (sum(ord(c) * (i + 1) for i, c in enumerate(name)) % 360) / 360
    r, g, b = colorsys.hsv_to_rgb(h, 0.65, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


def _font(size):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _outline(mask):
    """Boolean edge pixels of a mask."""
    m = mask.astype(bool)
    inner = m.copy()
    inner[1:-1, 1:-1] = m[1:-1, 1:-1] & m[:-2, 1:-1] & m[2:, 1:-1] & m[1:-1, :-2] & m[1:-1, 2:]
    edge = m & ~inner
    # thicken for visibility
    thick = edge.copy()
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)):
        thick |= np.roll(np.roll(edge, dy, 0), dx, 1)
    return thick


def draw_parts(image, parts):
    arr = np.asarray(image.convert("RGB")).astype(np.float32)
    labels = []
    for p in parts:
        c = np.array(_part_color(p["part"]), dtype=np.float32)
        m = p["mask"]
        arr[m] = 0.6 * arr[m] + 0.4 * c
        arr[_outline(m)] = c
        ys, xs = np.nonzero(m)
        if len(xs):
            labels.append((int(xs.mean()), int(ys.mean()), p["part"].replace("_", " "), tuple(int(v) for v in c)))
    out = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    d = ImageDraw.Draw(out)
    font = _font(max(14, out.width // 60))
    for x, y, text, c in labels:
        d.text((x, y), text, fill=(255, 255, 255), font=font, anchor="mm", stroke_width=3, stroke_fill=(0, 0, 0))
    return out


def draw_damage(image, damage, masks):
    arr = np.asarray(image.convert("RGB")).astype(np.float32)
    tags = []
    for i, (d, m) in enumerate(zip(damage, masks), start=1):
        c = np.array(ACTION_COLORS.get(d["repair_action"], (255, 255, 255)), dtype=np.float32)
        arr[m] = 0.55 * arr[m] + 0.45 * c
        arr[_outline(m)] = c
        b = d["bbox"]
        tags.append((b["x1"], b["y1"], f"#{i} {d['damage_type']}", tuple(int(v) for v in c)))
    out = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    dr = ImageDraw.Draw(out)
    font = _font(max(14, out.width // 55))
    for x, y, text, c in tags:
        dr.text((x + 4, max(0, y - 4)), text, fill=c, font=font, anchor="lb", stroke_width=3, stroke_fill=(0, 0, 0))
    return out


def analyze(image, vin, threshold):
    if image is None:
        return None, None, [], "Upload a photo first."
    image = image.convert("RGB")
    image.thumbnail((MAX_SIDE, MAX_SIDE))

    notes = []
    vin = (vin or "").strip()
    try:
        r = pipeline.analyze_image(image, vin=vin or None, damage_threshold=threshold, include_masks=True)
    except ValueError as e:  # invalid VIN: still analyze the photo
        notes.append(f"**VIN problem:** {e}")
        r = pipeline.analyze_image(image, vin=None, damage_threshold=threshold, include_masks=True)
    except Exception as e:  # e.g. NHTSA unreachable
        notes.append(f"**Couldn't decode VIN:** {e}")
        r = pipeline.analyze_image(image, vin=None, damage_threshold=threshold, include_masks=True)

    rows = [[i, d["damage_type"], f"{d['confidence']:.2f}",
             (d["part"] or "-").replace("_", " ") + (f" (+{', '.join(d['also_on'])})" if d["also_on"] else ""),
             f"{d['extent']:.0%} of part" if d["extent"] is not None else f"{d['image_fraction']:.1%} of photo",
             d["repair_action"], d["action_basis"]]
            for i, d in enumerate(r["damage"], start=1)]

    v = r["vehicle"]
    if v:
        notes.insert(0, f"**Vehicle (VIN):** {v['year']} {v['make']} {v['model']}"
                        + (f" {v['trim']}" if v["trim"] else "") + (f" · {v['body_style']}" if v["body_style"] else "")
                        + (f"  \n_NHTSA note: {v['warnings']}_" if v["warnings"] else ""))
    if not r["vehicle_found"]:
        notes.append("No vehicle detected in the photo.")
    if not rows:
        notes.append("No damage detected above the confidence threshold.")
    if any("collision" in d["action_basis"] for d in r["damage"]):
        notes.append("Repairs were escalated one level: the photo shows signs of a real collision "
                     "(broken lamp / shattered glass, or 4+ separate damages).")
    notes.append(f"**Parts detected:** {', '.join(p.replace('_', ' ') for p in r['parts_detected']) or 'none'}")

    return (draw_damage(image, r["damage"], r["damage_masks"]), draw_parts(image, r["parts"]),
            rows, "\n\n".join(notes))


def build_ui():
    with gr.Blocks(title="Autodamage") as demo:
        gr.Markdown("# Autodamage\nUpload a photo of a damaged car. Optionally add the VIN to identify the exact vehicle.")
        with gr.Row():
            with gr.Column(scale=1):
                img = gr.Image(type="pil", label="Photo")
                vin = gr.Textbox(label="VIN (optional)", placeholder="17 characters, e.g. 1HGCM82633A004352")
                thr = gr.Slider(0.3, 0.95, value=0.5, step=0.05, label="Damage confidence threshold",
                                info="Higher = fewer false alarms, more missed damage")
                btn = gr.Button("Analyze", variant="primary")
                if os.path.isdir(EXAMPLES_DIR):
                    ex = [[os.path.join(EXAMPLES_DIR, f)] for f in sorted(os.listdir(EXAMPLES_DIR))
                          if f.lower().endswith((".jpg", ".jpeg", ".png"))]
                    if ex:
                        gr.Examples(ex, inputs=[img], label="Test photos")
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("Damage + repair"):
                        out_dmg = gr.Image(label="Green = cosmetic, orange = repair, red = replace")
                    with gr.Tab("Parts"):
                        out_parts = gr.Image(label="Detected parts")
                notes = gr.Markdown()
                table = gr.Dataframe(headers=["#", "damage", "conf", "part", "extent", "repair", "why"],
                                     label="Damage list", wrap=True)
        btn.click(analyze, inputs=[img, vin, thr], outputs=[out_dmg, out_parts, table, notes])
    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part-checkpoint", default=None, help="part model (defaults to the pipeline's)")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    print("Loading models...")
    pipeline.load_models(part_checkpoint=args.part_checkpoint)
    build_ui().launch(server_name="127.0.0.1", server_port=args.port)


if __name__ == "__main__":
    main()
