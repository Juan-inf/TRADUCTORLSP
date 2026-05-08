"""
Script de ejecución completo del pipeline LSP.
Uso desde Colab o terminal:
  python scripts/run_pipeline.py --stage all --dataset_path /ruta/al/dataset
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    p = argparse.ArgumentParser(description="Pipeline LSP")
    p.add_argument("--stage", choices=["eda", "preprocess", "train", "eval", "export", "all"],
                   default="all")
    p.add_argument("--dataset_path", type=str, default="data/raw",
                   help="Ruta raíz del dataset de videos MP4")
    p.add_argument("--model",        type=str, default="fusion",
                   choices=["cnn_lstm", "stgcn", "videomae", "fusion"])
    p.add_argument("--epochs",       type=int, default=50)
    p.add_argument("--batch_size",   type=int, default=8)
    p.add_argument("--device",       type=str, default="cuda")
    p.add_argument("--output_dir",   type=str, default="checkpoints")
    p.add_argument("--wandb",        action="store_true")
    return p.parse_args()


def stage_eda(args):
    print("\n[1/5] EDA del dataset...")
    import numpy as np
    import pandas as pd
    import cv2
    from tqdm import tqdm

    dataset_path = Path(args.dataset_path)
    records = []

    for video_path in tqdm(list(dataset_path.rglob("*.mp4"))):
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            ret, _ = cap.read()
            records.append({
                "ruta":  str(video_path),
                "clase": video_path.parent.name,
                "n_frames": n,
                "fps": fps,
                "ancho": w, "alto": h,
                "duracion_seg": n / max(fps, 1),
                "valido": ret,
            })
        cap.release()

    df = pd.DataFrame(records)
    os.makedirs("data", exist_ok=True)
    df[df["valido"]].to_csv("data/manifest.csv", index=False)

    classes = sorted(df["clase"].unique())
    label2idx = {c: i for i, c in enumerate(classes)}

    with open("data/label2idx.json", "w", encoding="utf-8") as f:
        json.dump(label2idx, f, ensure_ascii=False, indent=2)

    counts = df["clase"].value_counts()
    n_cls  = len(classes)
    weights = {label2idx[c]: len(df) / (n_cls * cnt) for c, cnt in counts.items()}

    with open("data/class_weights.json", "w") as f:
        json.dump({str(k): v for k, v in weights.items()}, f, indent=2)

    print(f"  Videos válidos: {df['valido'].sum()}")
    print(f"  Clases: {n_cls}")
    print(f"  Manifest: data/manifest.csv")
    return df


def stage_preprocess(args):
    print("\n[2/5] Preprocesamiento de videos y landmarks...")
    from src.preprocessing.video_preprocessor import VideoPreprocessor
    from src.preprocessing.landmark_extractor import LandmarkExtractor
    from src.dataset.lsp_dataset import create_splits

    create_splits("data/manifest.csv", "data/manifest_splits.csv")

    preprocessor = VideoPreprocessor(n_frames=30, imagenet_norm=True)
    preprocessor.batch_process(
        manifest_csv="data/manifest_splits.csv",
        output_dir="data/processed_pixels",
    )

    extractor = LandmarkExtractor(use_pose=True, use_face=False)
    extractor.batch_extract(
        manifest_csv="data/manifest_splits.csv",
        output_dir="data/landmarks",
        n_frames=30,
    )
    print("  Preprocesamiento completado.")


def stage_train(args):
    print(f"\n[3/5] Entrenando modelo: {args.model}...")
    import torch
    from src.models import CNNLSTM, STGCN, VideoMAEWrapper, LSPFusionModel
    from src.dataset.lsp_dataset import get_dataloaders
    from src.training.trainer import LSPTrainer

    with open("data/label2idx.json") as f:
        label2idx = json.load(f)
    with open("data/class_weights.json") as f:
        cw = json.load(f)

    n_classes = len(label2idx)

    mode_map = {
        "cnn_lstm": "pixels",
        "stgcn":    "landmarks",
        "videomae": "pixels",
        "fusion":   "both",
    }
    mode = mode_map[args.model]

    loaders = get_dataloaders(
        manifest_csv="data/manifest_splits.csv",
        label2idx=label2idx,
        mode=mode,
        pixels_dir="data/processed_pixels",
        landmarks_dir="data/landmarks",
        batch_size=args.batch_size,
        num_workers=4,
        class_weights={int(k): v for k, v in cw.items()},
    )

    if args.model == "cnn_lstm":
        model = CNNLSTM(n_classes=n_classes, pretrained=True)
    elif args.model == "stgcn":
        model = STGCN(n_classes=n_classes, n_nodes=75)
    elif args.model == "videomae":
        model = VideoMAEWrapper(n_classes=n_classes, pretrained=True)
    else:
        pixel_bb    = CNNLSTM(n_classes=n_classes, pretrained=True)
        landmark_bb = STGCN(n_classes=n_classes, n_nodes=75)
        model = LSPFusionModel(
            pixel_backbone=pixel_bb, landmark_backbone=landmark_bb,
            dim_pixels=1024, dim_landmarks=256,
            n_classes=n_classes, fusion_strategy="concat",
        )

    cw_tensor = torch.tensor([cw.get(str(i), 1.0) for i in range(n_classes)], dtype=torch.float32)

    trainer = LSPTrainer(
        model=model, loaders=loaders, mode=mode,
        n_classes=n_classes, device=args.device,
        output_dir=args.output_dir,
        lr=1e-4, epochs=args.epochs, patience=10,
        class_weights=cw_tensor,
        wandb_project="traductor-lsp" if args.wandb else None,
        model_name=args.model,
    )

    history = trainer.train()
    return trainer


def stage_eval(args, trainer=None):
    print("\n[4/5] Evaluando en test set...")
    if trainer is None:
        print("  Cargar checkpoint manualmente para evaluar.")
        return
    ckpt = f"{args.output_dir}/{args.model}_best.pt"
    metrics = trainer.evaluate_test(ckpt)
    print(f"  Test accuracy: {metrics.get('accuracy', 0):.4f}")
    print(f"  Test F1-macro: {metrics.get('f1_macro', 0):.4f}")


def stage_export(args, trainer=None):
    print("\n[5/5] Exportando a ONNX...")
    if trainer is None:
        print("  Cargar checkpoint para exportar.")
        return
    ckpt = f"{args.output_dir}/{args.model}_best.pt"
    out  = f"{args.output_dir}/lsp_model.onnx"
    trainer.export_onnx(ckpt, out)


if __name__ == "__main__":
    args = parse_args()

    if args.stage in ("eda", "all"):
        stage_eda(args)
    if args.stage in ("preprocess", "all"):
        stage_preprocess(args)
    trainer = None
    if args.stage in ("train", "all"):
        trainer = stage_train(args)
    if args.stage in ("eval", "all"):
        stage_eval(args, trainer)
    if args.stage in ("export", "all"):
        stage_export(args, trainer)

    print("\nPipeline completado.")
