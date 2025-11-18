import os
import base64
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.colors as mcolors
from scipy import ndimage
from openai import OpenAI

# ============================================================
# CONFIG
# ============================================================

INPUT_PANEL = "results_full_finetune_8e_gradcam/Pleural_Effusion/Pleural_Effusion_extra_TP_0.png"
CLASS_NAME = "Pleural Effusion"

OUTPUT_ROOT = "results_full_finetune_8e_gradcam_itps"
CASE_NAME = "pleural_eff_TP0"

INTENSITY_V_THRESH = 0.15
INTENSITY_S_THRESH = 0.1
MIN_REGION_FRAC = 0.005

BASELINE_BOX_COLOR = "red"
FINETUNED_BOX_COLOR = "red"
BOX_WIDTH = 3


# ============================================================
# HELPERS
# ============================================================

def load_panel(path):
    img = Image.open(path).convert("RGB")
    return img, np.array(img)


def split_quadrants(panel_np):
    H, W, _ = panel_np.shape
    mid_y, mid_x = H // 2, W // 2
    qh, qw = mid_y, mid_x

    title_frac = 0.15

    bo_y0 = int(title_frac * qh)
    bo_y1 = qh
    bo_x0 = 0
    bo_x1 = qw

    fo_y0 = mid_y + int(title_frac * qh)
    fo_y1 = H
    fo_x0 = 0
    fo_x1 = qw

    crops = {
        "baseline": panel_np[bo_y0:bo_y1, bo_x0:bo_x1, :],
        "finetuned": panel_np[fo_y0:fo_y1, fo_x0:fo_x1, :],
    }
    locs = {
        "baseline": (bo_x0, bo_y0, bo_x1, bo_y1),
        "finetuned": (fo_x0, fo_y0, fo_x1, fo_y1),
    }
    return crops, locs


def detect_heat_boxes(overlay_np):
    img_float = overlay_np.astype(np.float32) / 255.0
    hsv = mcolors.rgb_to_hsv(img_float)
    Hc, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    mask = (V > INTENSITY_V_THRESH) & (S > INTENSITY_S_THRESH)

    hot_mask = (Hc < 0.25) | (Hc > 0.95)
    mask &= hot_mask

    if mask.sum() == 0:
        return []

    labels, num = ndimage.label(mask)
    Hh, Wh = mask.shape
    min_pixels = int(MIN_REGION_FRAC * Hh * Wh)

    boxes = []
    for lab in range(1, num + 1):
        ys, xs = np.where(labels == lab)
        if ys.size == 0:
            continue
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        area = (x1 - x0 + 1) * (y1 - y0 + 1)
        if area < min_pixels:
            continue
        boxes.append((int(x0), int(y0), int(x1), int(y1)))

    return boxes


def draw_boxes_on_panel(panel_pil, locs, boxes_baseline, boxes_finetuned):
    draw = ImageDraw.Draw(panel_pil)

    bo_x0, bo_y0, _, _ = locs["baseline"]
    for (x0, y0, x1, y1) in boxes_baseline:
        panel_box = (bo_x0 + x0, bo_y0 + y0, bo_x0 + x1, bo_y0 + y1)
        draw.rectangle(panel_box, outline=BASELINE_BOX_COLOR, width=BOX_WIDTH)

    fo_x0, fo_y0, _, _ = locs["finetuned"]
    for (x0, y0, x1, y1) in boxes_finetuned:
        panel_box = (fo_x0 + x0, fo_y0 + y0, fo_x0 + x1, fo_y0 + y1)
        draw.rectangle(panel_box, outline=FINETUNED_BOX_COLOR, width=BOX_WIDTH)

    return panel_pil


def encode_image_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def call_openai_vision(annotated_png_path, class_name, out_txt_path):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("<<NO API KEY FOUND>>")
        return

    client = OpenAI(api_key=api_key)
    img_b64 = encode_image_base64(annotated_png_path)

    prompt = f"""
Act as a professional radiologist and AI model-interpretability expert.

You are given a Grad-CAM panel showing:
- Top-left: Baseline model heatmap overlay
- Bottom-left: Full-parameter finetuned model heatmap overlay
- Bounding boxes drawn around detected high-activation regions.

Task:
1. State whether the boxed regions are actually relevant to the pathology classified: {class_name}.
2. Make a judgment on whether the bounding boxes reflect correct or incorrect reasoning behind the model's prediction.
3. If relevant and correct, explain what details in (or around) the boxed regions actually present in this specific x-ray case
have led to the correct prediction (don't just say the bounding boxes are in correct location, tell what details are visually
present in those locations indicative of the predicted pathology). If relevant and incorrect, explain why incorrect. If
irrelevant, explain why the model may have gone with an irrelevant activation area.
"""

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                    }
                ],
            }
        ],
        max_tokens=600,
        temperature=0.4,
    )

    text = resp.choices[0].message.content.strip()
    with open(out_txt_path, "w") as f:
        f.write(text)


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    case_dir = os.path.join(OUTPUT_ROOT, CASE_NAME)
    os.makedirs(case_dir, exist_ok=True)

    panel_pil, panel_np = load_panel(INPUT_PANEL)
    crops, locs = split_quadrants(panel_np)

    bo = crops["baseline"]
    fo = crops["finetuned"]

    boxes_b = detect_heat_boxes(bo)
    boxes_f = detect_heat_boxes(fo)

    annotated = draw_boxes_on_panel(panel_pil.copy(), locs, boxes_b, boxes_f)

    out_img_path = os.path.join(case_dir, "annotated_panel.png")
    annotated.save(out_img_path, dpi=(200, 200))

    out_txt_path = os.path.join(case_dir, "gpt_analysis.txt")
    call_openai_vision(out_img_path, CLASS_NAME, out_txt_path)

    print("Saved:")
    print(" -", out_img_path)
    print(" -", out_txt_path)


if __name__ == "__main__":
    main()
