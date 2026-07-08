# Fine-Tuning SegFormer for Land-Cover Segmentation over the Peak District

MSc thesis project (Advanced GIS and Remote Sensing, Cranfield University). Fine-tunes a pretrained SegFormer-B0 semantic segmentation model on Sentinel-derived imagery patches of the Peak District, evaluating whether a transformer-based foundation model can be adapted to a fine-grained land-cover classification task with a small, task-specific dataset.

## Motivation

Foundation models trained on general-purpose segmentation benchmarks (e.g. ADE20K) don't natively recognise domain-specific land-cover classes. This project asks: starting from a SegFormer-B0 checkpoint with zero task-specific training, how much can targeted fine-tuning on a modestly sized regional dataset close that gap — and how does the result compare to a purpose-built CNN baseline?

## Method

- **Dataset**: 1,027 image patches covering the Peak District, split into train/val/test via a fixed JSON split file, with per-pixel land-cover labels
- **Model**: SegFormer-B0 (Hugging Face `transformers`), evaluated first zero-shot, then fine-tuned
- **Training**: staged approach — a quick 2-epoch subset run to validate the pipeline, followed by a full run across 11 further epochs on the complete training set
- **Environment**: PyTorch, Hugging Face `transformers` + `accelerate`, Google Colab (GPU runtime)
- **Evaluation**: pixel accuracy and mean IoU (mIoU) computed via confusion matrix on the held-out test set, tracked at both the zero-shot stage and after fine-tuning, alongside qualitative triptych visualisations (input / ground truth / prediction) for the top-performing classes

## Results

| Metric | Zero-shot | Fine-tuned |
|---|---|---|
| Pixel accuracy | 1.39% | 36.44% |
| Mean IoU | 0.07% | 5.15% |

Fine-tuning produced a substantial jump in both metrics over the zero-shot baseline, confirming the model adapted meaningfully to the domain from a limited number of epochs and a relatively small dataset.

**Limitation, stated plainly**: a purpose-built CNN benchmark (Van der Plas et al.) still outperforms the fine-tuned SegFormer-B0 on this task. Class imbalance across the land-cover categories constrained mIoU gains in particular — some minority classes remained poorly segmented even after fine-tuning. The result is best read as evidence that transformer-based foundation models can be adapted to this kind of task with limited compute and data, not as a claim that they beat specialised architectures out of the box.

## Repository structure

```
.
├── notebooks/
│   └── PeakDistrict_EndToEnd_Clean.ipynb   # full pipeline: data loading, zero-shot eval, fine-tuning, fine-tuned eval
├── figures/                                 # before/after prediction patches, accuracy & mIoU progression charts
└── README.md
```

## Running it

The notebook is designed to run end-to-end in Google Colab with a GPU runtime:

1. Upload the dataset (patches + `lc_label_names.json` + the `train_test_split_80tiles*.json` split file) to the Colab session, or mount Google Drive
2. Run the setup cell to install dependencies (`transformers`, `accelerate`, `scikit-learn`) and load the model
3. Run the zero-shot evaluation section to reproduce the baseline metrics and triptychs
4. Run the fine-tuning section (quick 2-epoch validation run, then the full 12-epoch run) to reproduce the fine-tuned results
5. Outputs (predictions, metrics, figures) are written to `/content/predictions_zero_shot` and `/content/predictions_fine_tuned`

## Author

Kumbiraishe Chagonda — [LinkedIn](https://www.linkedin.com/in/kumbiraishe-chagonda-15a3a9271/)
