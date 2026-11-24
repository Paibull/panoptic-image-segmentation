import csv
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


ROOT = Path("./").resolve()
IMAGE_DIR = ROOT / "data" / "indoor-40-test.v2i.coco" / "train"
ANNOTATION_FILE = IMAGE_DIR / "_annotations.coco.json"
CLASS_NAMES_FILE = ROOT / "configs" / "class_names.json"
CHECKPOINT_DIR = ROOT / "weights"
OUTPUT_DIR = ROOT / "outputs" / "inference_40"
CONFIDENCE_THRESHOLD = 0.25
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINTS = [
    ("yolo26n-seg", "pretrained", "yolo", CHECKPOINT_DIR / "yolo26n-seg/pretrained/best.pt"),
    ("yolo26n-seg", "scratch", "yolo", CHECKPOINT_DIR / "yolo26n-seg/scratch/best.pt"),
    ("mask-rcnn", "pretrained", "mask-rcnn", CHECKPOINT_DIR / "mask-rcnn/pretrained/best.ckpt"),
    ("mask-rcnn", "scratch", "mask-rcnn", CHECKPOINT_DIR / "mask-rcnn/scratch/best.ckpt"),
    ("rfdetr-seg-nano", "pretrained", "rfdetr", CHECKPOINT_DIR / "rfdetr-seg-nano/pretrained/checkpoint_best_total.pth"),
    ("rfdetr-seg-nano", "scratch", "rfdetr", CHECKPOINT_DIR / "rfdetr-seg-nano/scratch/checkpoint_best_total.pth"),
]


def check_inputs():
    required = [ANNOTATION_FILE, CLASS_NAMES_FILE, *(item[3] for item in CHECKPOINTS)]
    missing = [path for path in required if not path.is_file()]
    if not IMAGE_DIR.is_dir():
        missing.append(IMAGE_DIR)

    if missing:
        paths = "\n".join(f"- {path.relative_to(ROOT)}" for path in missing)
        raise FileNotFoundError(
            "Copy the COCO export and six checkpoints into these paths, then run from the repo root:\n"
            + paths
        )


def build_mask_rcnn(num_classes):
    model = maskrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
        trainable_backbone_layers=5,
    )
    box_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(box_features, num_classes)
    mask_features = model.roi_heads.mask_predictor.conv5_mask.in_channels
    mask_hidden = model.roi_heads.mask_predictor.conv5_mask.out_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        mask_features, mask_hidden, num_classes
    )
    return model


def load_model(kind, checkpoint_path, class_names):
    if kind == "yolo":
        from ultralytics import YOLO

        return YOLO(str(checkpoint_path)), class_names

    if kind == "mask-rcnn":
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        checkpoint_names = checkpoint.get("class_names", class_names)
        if checkpoint_names != class_names:
            raise ValueError("Mask R-CNN class names differ from configs/class_names.json.")
        model = build_mask_rcnn(
            checkpoint.get("num_classes", len(class_names) + 1)
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        return model.to(DEVICE).eval(), checkpoint_names

    from rfdetr import from_checkpoint

    return from_checkpoint(str(checkpoint_path)), class_names


def resize_mask(mask, height, width):
    mask = np.asarray(mask).astype(np.uint8)
    if mask.shape != (height, width):
        mask = np.asarray(
            Image.fromarray(mask).resize((width, height), Image.Resampling.NEAREST)
        )
    return mask > 0


def predict(model, kind, image_path, image, class_names):
    height, width = np.asarray(image).shape[:2]
    predictions = []

    if kind == "yolo":
        result = model.predict(
            str(image_path),
            conf=CONFIDENCE_THRESHOLD,
            retina_masks=True,
            verbose=False,
        )[0]
        if result.masks is None:
            return predictions

        for mask, label, score in zip(
            result.masks.data.cpu().numpy(),
            result.boxes.cls.cpu().numpy().astype(int),
            result.boxes.conf.cpu().numpy(),
        ):
            predictions.append(
                {
                    "mask": resize_mask(mask, height, width),
                    "class_name": str(result.names[int(label)]),
                    "class_id": int(label),
                    "confidence": float(score),
                }
            )
        return predictions

    if kind == "mask-rcnn":
        image_tensor = torch.from_numpy(np.asarray(image).copy())
        image_tensor = image_tensor.permute(2, 0, 1).float().to(DEVICE) / 255
        with torch.inference_mode():
            result = model([image_tensor])[0]

        for mask, label, score in zip(
            result["masks"][:, 0].cpu().numpy(),
            result["labels"].cpu().numpy().astype(int),
            result["scores"].cpu().numpy(),
        ):
            class_index = int(label) - 1
            if score < CONFIDENCE_THRESHOLD or not 0 <= class_index < len(class_names):
                continue
            predictions.append(
                {
                    "mask": mask >= 0.5,
                    "class_name": class_names[class_index],
                    "class_id": class_index,
                    "confidence": float(score),
                }
            )
        return predictions

    from rfdetr import ClassThresholds

    result = model.predict(
        str(image_path),
        threshold=ClassThresholds({}, default=CONFIDENCE_THRESHOLD),
    )
    if not len(result) or result.mask is None:
        return predictions

    for mask, label, score in zip(
        np.asarray(result.mask),
        np.asarray(result.class_id).astype(int),
        np.asarray(result.confidence),
    ):
        if not 0 <= label < len(class_names):
            continue
        predictions.append(
            {
                "mask": resize_mask(mask, height, width),
                "class_name": class_names[label],
                "class_id": int(label),
                "confidence": float(score),
            }
        )
    return predictions


def save_overlay(image, predictions, output_path):
    colors = [
        (230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48),
        (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60),
        (250, 190, 190), (0, 128, 128), (230, 190, 255), (170, 110, 40),
    ]
    output = np.asarray(image).copy()

    for item in predictions:
        mask = resize_mask(item["mask"], output.shape[0], output.shape[1])
        color = np.asarray(colors[item["class_id"] % len(colors)])
        output[mask] = (0.45 * output[mask] + 0.55 * color).astype(np.uint8)

    rendered = Image.fromarray(output)
    draw = ImageDraw.Draw(rendered)
    for index, item in enumerate(predictions[:20]):
        label = f"{item['class_name']} {item['confidence']:.2f}"
        draw.text(
            (8, 8 + index * 18),
            label,
            fill="white",
            stroke_width=1,
            stroke_fill="black",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered.save(output_path)


def main():
    check_inputs()
    class_names = json.loads(CLASS_NAMES_FILE.read_text(encoding="utf-8"))
    images = json.loads(ANNOTATION_FILE.read_text(encoding="utf-8"))["images"]
    prediction_rows = []
    timing_rows = []

    print(f"Images: {len(images)} | device: {DEVICE}")

    for model_name, weights, kind, checkpoint_path in CHECKPOINTS:
        model, model_class_names = load_model(kind, checkpoint_path, class_names)
        output_dir = OUTPUT_DIR / model_name / weights
        start_time = time.perf_counter()
        detection_count = 0

        for image_info in images:
            image_path = IMAGE_DIR / image_info["file_name"]
            with Image.open(image_path) as opened_image:
                image = opened_image.convert("RGB")

            predictions = predict(
                model, kind, image_path, image, model_class_names
            )
            save_overlay(image, predictions, output_dir / image_path.name)
            detection_count += len(predictions)
            prediction_rows.append(
                {
                    "model": model_name,
                    "weights": weights,
                    "image": image_path.name,
                    "detections": len(predictions),
                    "classes": "; ".join(item["class_name"] for item in predictions),
                }
            )

        elapsed = time.perf_counter() - start_time
        timing_rows.append(
            {
                "model": model_name,
                "weights": weights,
                "images": len(images),
                "detections": detection_count,
                "elapsed_seconds": round(elapsed, 2),
                "seconds_per_image": round(elapsed / max(len(images), 1), 2),
            }
        )
        print(f"{model_name} {weights}: {elapsed:.1f}s")

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename, rows in [
        ("predictions.csv", prediction_rows),
        ("inference_times.csv", timing_rows),
    ]:
        with (OUTPUT_DIR / filename).open(
            "w", newline="", encoding="utf-8"
        ) as output_file:
            writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    print(f"Saved results to {OUTPUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
