# Segmentasi Instance Indoor - Visi Komputer

Fa'iz Akbar Hizbullah - 5054241005

Repository ini berisi eksperimen segmentasi instance objek indoor menggunakan RF-DETR-Seg Nano, YOLO26n-Seg, dan Mask R-CNN ResNet50-FPN.

Model dibandingkan dengan bobot pretrained dan scratch. Setiap run menggunakan 50 epoch. Evaluasi memakai test set resmi dan 40 foto personal yang tetap dipisahkan dari data training dan validasi.

## Dataset

Split resmi terdiri dari 397 gambar train, 166 validasi, dan 359 test. Foto personal digunakan sebagai test set tambahan. Dataset tersedia di [Panoptic Indoor Segmentation, Roboflow](https://universe.roboflow.com/panopticindoor/panoptic-indoor-segmentation/dataset/5).

## Struktur repository

```text
.
├── configs/
│   └── class_names.json
├── deklarasi_AI.md
├── notebooks/
│   ├── inference_40_images.py
│   ├── mask-rcnn.ipynb
│   ├── rfdetr-seg-nano.ipynb
│   └── yolo26n-seg.ipynb
├── reports/
│   ├── annotation_evidence/
│   ├── failure_study_v2/
│   ├── inference_metrics/
│   ├── training_logs_for_submission/
│   ├── image_characteristics_per_image.csv
│   ├── original_vs_personal_test_mask_ap.png
│   └── pretrained_vs_scratch_metrics.png
├── logbook.md
├── requirements-inference.txt
└── README.md
```

## Catatan

Notebook training dijalankan di Kaggle. Dataset, checkpoint model, dan foto personal tidak disertakan di repository.
