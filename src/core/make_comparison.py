#!/usr/bin/env python
"""Generate side-by-side comparison images of all 5 pipelines for given scenarios."""

import sys
import os
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "outputs", "msr50_eval"
)
FRAMES_DIR = os.path.join(BASE_DIR, "frames")

PIPELINES = ["Ours", "StoryDiffusion", "Markovian", "NoBridge", "GlobalInject"]

LABEL_WIDTH = 320
FONT_SIZE = 48
BG_COLOR = (30, 30, 30)
TEXT_COLOR = (255, 255, 255)
ROW_GAP = 6


def get_font(size):
    """Try to load a readable font, fall back to default."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-Bold.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def make_comparison(scenario_id: str):
    """Create a comparison image for a single scenario."""
    # Load all contact sheets
    sheets = {}
    for pipe in PIPELINES:
        path = os.path.join(FRAMES_DIR, pipe, scenario_id, "contact_sheet.png")
        if not os.path.exists(path):
            print(f"  WARNING: missing {path}")
            continue
        sheets[pipe] = Image.open(path).convert("RGB")

    if not sheets:
        print(f"  No contact sheets found for {scenario_id}, skipping.")
        return

    # Determine target width (use max width among sheets)
    target_w = max(img.width for img in sheets.values())

    # Resize sheets to match target width (preserve aspect ratio)
    resized = {}
    for pipe, img in sheets.items():
        if img.width != target_w:
            scale = target_w / img.width
            new_h = int(img.height * scale)
            resized[pipe] = img.resize((target_w, new_h), Image.LANCZOS)
        else:
            resized[pipe] = img

    # Calculate total canvas size
    total_w = LABEL_WIDTH + target_w
    total_h = sum(resized[p].height for p in PIPELINES if p in resized)
    total_h += ROW_GAP * (len(resized) - 1)

    canvas = Image.new("RGB", (total_w, total_h), BG_COLOR)
    draw = ImageDraw.Draw(canvas)
    font = get_font(FONT_SIZE)

    y = 0
    for pipe in PIPELINES:
        if pipe not in resized:
            continue
        img = resized[pipe]

        # Draw label strip
        label_region = (0, y, LABEL_WIDTH, y + img.height)
        draw.rectangle(label_region, fill=BG_COLOR)

        # Center text vertically in the label strip
        bbox = font.getbbox(pipe)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        tx = (LABEL_WIDTH - tw) // 2
        ty = y + (img.height - th) // 2
        draw.text((tx, ty), pipe, fill=TEXT_COLOR, font=font)

        # Paste contact sheet
        canvas.paste(img, (LABEL_WIDTH, y))

        y += img.height + ROW_GAP

    out_path = os.path.join(BASE_DIR, f"comparison_{scenario_id}.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canvas.save(out_path, quality=95)
    print(f"  Saved: {out_path}")


def main():
    if len(sys.argv) > 1:
        scenarios = sys.argv[1:]
    else:
        scenarios = ["scifi_01", "scifi_02", "scifi_03"]

    for sid in scenarios:
        print(f"Generating comparison for {sid}...")
        make_comparison(sid)

    print("Done.")


if __name__ == "__main__":
    main()
