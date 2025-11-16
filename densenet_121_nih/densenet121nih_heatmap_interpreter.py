import os
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.colors as mcolors
from scipy import ndimage
from openai import OpenAI

# ==========================
# CONFIG
# ==========================

INPUT_PANEL = "results_full_finetune_gradcam/Pleural_Effusion/Pleural_Effusion_3.png"

CLASS_NAME = "Pleural Effusion"

OUTPUT_ROOT = "results_full_finetune_gradcam_itps"
CASE_NAME = "pleural_eff_3"

INTENSITY_V_THRESH = 0.55
INTENSITY_S_THRESH = 0.35
MIN_REGION_FRAC = 0.005

BASELINE_BOX_COLOR = "cyan"
FINETUNED_BOX_COLOR = "red"
BOX_WIDTH = 3

# ==========================
# HELPER FUNCTIONS
# ==========================

def load_panel(path):
    img = Image.open(path).convert("RGB")
    return img, np.array(img)


def split_quadrants(panel_np):
    """
    Assuming a 2x2 layout with roughly equal quadrants:
    [ Baseline Overlay | Baseline Raw ]
    [ Finetuned Overlay | Finetuned Raw ]
    We cut by half height & width, then trim a little title margin on top.
    """
    H, W, _ = panel_np.shape
    mid_y, mid_x = H // 2, W // 2

    q_h, q_w = mid_y, mid_x

    title_frac = 0.15

    bo_y0 = int(0.0 * q_h + title_frac * q_h)
    bo_y1 = q_h
    bo_x0 = 0
    bo_x1 = q_w

    fo_y0 = mid_y + int(title_frac * q_h)
    fo_y1 = H
    fo_x0 = 0
    fo_x1 = q_w

    baseline_overlay = panel_np[bo_y0:bo_y1, bo_x0:bo_x1, :]
    finetuned_overlay = panel_np[fo_y0:fo_y1, fo_x0:fo_x1, :]

    locs = {
        "baseline": (bo_x0, bo_y0, bo_x1, bo_y1),
        "finetuned": (fo_x0, fo_y0, fo_x1, fo_y1),
    }
    crops = {
        "baseline": baseline_overlay,
        "finetuned": finetuned_overlay,
    }
    return crops, locs


def detect_heat_boxes(overlay_np,
                      v_thresh=INTENSITY_V_THRESH,
                      s_thresh=INTENSITY_S_THRESH,
                      min_region_frac=MIN_REGION_FRAC):
    """
    Given an RGB overlay image as numpy array [H,W,3], detect heat regions by:
      - converting to HSV
      - thresholding on value and saturation
      - focusing on hot (red/yellow/green) hues (not dark blue)
      - finding connected components and returning bounding boxes

    Returns: list of boxes [(x0,y0,x1,y1), ...] in overlay pixel coords.
    """
    img_float = overlay_np.astype(np.float32) / 255.0
    hsv = mcolors.rgb_to_hsv(img_float)
    Hc, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    mask = (V > v_thresh) & (S > s_thresh)

    hot_mask = (Hc < 0.25) | (Hc > 0.95)
    mask = mask & hot_mask

    Hh, Wh = mask.shape
    if mask.sum() == 0:
        return []

    labels, num = ndimage.label(mask)
    boxes = []
    min_pixels = int(min_region_frac * Hh * Wh)

    for lab in range(1, num + 1):
        ys, xs = np.where(labels == lab)
        if ys.size == 0:
            continue
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        area = (y1 - y0 + 1) * (x1 - x0 + 1)
        if area < min_pixels:
            continue
        boxes.append((int(x0), int(y0), int(x1), int(y1)))

    return boxes


def draw_boxes_on_panel(panel_pil, locs, boxes_baseline, boxes_finetuned):
    draw = ImageDraw.Draw(panel_pil)

    bo_x0, bo_y0, bo_x1, bo_y1 = locs["baseline"]
    for (x0, y0, x1, y1) in boxes_baseline:
        panel_box = (bo_x0 + x0, bo_y0 + y0, bo_x0 + x1, bo_y0 + y1)
        draw.rectangle(panel_box, outline=BASELINE_BOX_COLOR, width=BOX_WIDTH)

    fo_x0, fo_y0, fo_x1, fo_y1 = locs["finetuned"]
    for (x0, y0, x1, y1) in boxes_finetuned:
        panel_box = (fo_x0 + x0, fo_y0 + y0, fo_x0 + x1, fo_y0 + y1)
        draw.rectangle(panel_box, outline=FINETUNED_BOX_COLOR, width=BOX_WIDTH)

    return panel_pil


def summarize_boxes_text(boxes, width, height):
    if not boxes:
        return "no clearly localized high-activation regions"

    descs = []
    for (x0, y0, x1, y1) in boxes:
        w = width
        h = height
        cx = (x0 + x1) / 2.0 / w
        cy = (y0 + y1) / 2.0 / h
        area = (x1 - x0 + 1) * (y1 - y0 + 1) / (w * h)

        if cx < 0.33:
            horiz = "left"
        elif cx > 0.66:
            horiz = "right"
        else:
            horiz = "central"

        if cy < 0.33:
            vert = "upper"
        elif cy > 0.66:
            vert = "lower"
        else:
            vert = "middle"

        if area < 0.05:
            size = "small"
        elif area < 0.15:
            size = "medium"
        else:
            size = "large"

        descs.append(f"a {size} region in the {vert}-{horiz} chest")

    if len(descs) == 1:
        return descs[0]
    else:
        return "; ".join(descs)


def call_openai_summary(class_name, baseline_desc, finetuned_desc, out_path):
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None or api_key.strip() == "":
        summary = (
            "OPENAI_API_KEY not set; cannot call API.\n"
            "Baseline focus: " + baseline_desc + "\n"
            "Finetuned focus: " + finetuned_desc + "\n"
        )
        with open(out_path, "w") as f:
            f.write(summary)
        print("WARNING: OPENAI_API_KEY not found. Wrote fallback text summary.")
        return

    client = OpenAI(api_key=api_key)

    prompt = f"""
You are analyzing Grad-CAM heatmaps for chest X-rays.

Pathology: {class_name}

The heatmaps show where the model focuses inside the lungs.

- Baseline model heat regions: {baseline_desc}.
- Finetuned model heat regions: {finetuned_desc}.

Write a brief, 3 to 5 sentence summary comparing the two models' attention.
Explain qualitatively where each model is looking (e.g. upper vs lower, central vs peripheral),
whether the finetuned model seems more focused or more diffuse, and how that might relate
to false positives or false negatives for this pathology. Avoid mentioning coordinates or
pixel values; just describe the relative locations in anatomical terms.
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=250,
        temperature=0.4,
    )

    text = resp.choices[0].message.content.strip()
    with open(out_path, "w") as f:
        f.write(text)


# ==========================
# MAIN
# ==========================
def main():
    panel_pil, panel_np = load_panel(INPUT_PANEL)

    crops, locs = split_quadrants(panel_np)

    bo = crops["baseline"]
    fo = crops["finetuned"]

    boxes_baseline = detect_heat_boxes(bo)
    boxes_finetuned = detect_heat_boxes(fo)

    print(f"Detected {len(boxes_baseline)} baseline boxes, "
          f"{len(boxes_finetuned)} finetuned boxes.")

    annotated = draw_boxes_on_panel(panel_pil.copy(), locs,
                                    boxes_baseline, boxes_finetuned)

    case_dir = os.path.join(OUTPUT_ROOT, CASE_NAME)
    os.makedirs(case_dir, exist_ok=True)

    out_img_path = os.path.join(case_dir, "annotated_panel.png")
    out_txt_path = os.path.join(case_dir, "analysis.txt")

    annotated.save(out_img_path, dpi=(200, 200))
    print(f"Saved annotated panel to: {out_img_path}")

    h_bo, w_bo, _ = bo.shape
    h_fo, w_fo, _ = fo.shape

    baseline_desc = summarize_boxes_text(boxes_baseline, w_bo, h_bo)
    finetuned_desc = summarize_boxes_text(boxes_finetuned, w_fo, h_fo)

    call_openai_summary(CLASS_NAME, baseline_desc, finetuned_desc, out_txt_path)
    print(f"Saved text summary to: {out_txt_path}")


if __name__ == "__main__":
    main()
