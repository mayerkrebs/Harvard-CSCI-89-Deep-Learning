# Microscopic Wood Genus Classification Using Convolutional Neural Networks and Transfer Learning

## Project overview

This project classifies transverse microscopic wood images from XDD_015 into 17 genera. The model uses EfficientNetV2B0 pretrained on ImageNet with a small custom convolutional head. The head is trained first with the backbone frozen, followed by limited fine-tuning.

All images from one specimen stay in the same split. This avoids leakage between related images of the same wood sample. A separate notebook compares learned feature activations with genus-level wood-anatomy characteristics based on the IAWA hardwood-feature framework.

## Main results

Results for the final fine-tuned model:

| Evaluation unit | Test support | Accuracy | Top-2 accuracy | Macro F1 |
|---|---:|---:|---:|---:|
| Specimen | 69 specimens | 0.884 | 0.957 | 0.864 |
| Image | 893 images | 0.873 | 0.944 | 0.839 |

## Repository structure

- `1. data_pipeline.ipynb`: filters the dataset, splits by specimen, and exports PNG files and a manifest.
- `2. train_model.ipynb`: trains, fine-tunes, and evaluates the model.
- `3. feature_interpretation.ipynb`: compares learned activations with wood-anatomy characteristics.
- `genus_characteristics.csv`: genus-level characteristics used in the interpretation notebook.
- `create_sample_dataset.py` and `sample_data/`: create and store a small example of the processed-data format.
- `Project Report.md` and `report_assets/`: full report and figures.
- `requirements.txt`: Python dependencies used for the project.

## Dataset

The data come from [XDD_015 in the Kyoto University Research Information Repository](https://repository.kulib.kyoto-u.ac.jp/items/7f0f28bd-f9b0-4603-be8e-7c7d908a245f). The source file is `WIG_v1.2.1_600.h5`.

| Dataset stage | Genera | Specimens | Images |
|---|---:|---:|---:|
| Raw archive | 33 | 540 | 7,051 |
| Retained dataset | 17 | 466 | 5,992 |
| Training split | 17 | 328 | 4,263 |
| Validation split | 17 | 69 | 836 |
| Test split | 17 | 69 | 893 |

Genera with fewer than 12 specimens were removed. Splits were made by specimen within each genus, so images from the same wood sample cannot appear in more than one split.

The first notebook reads the HDF5 archive once, exports grayscale PNG files to `wood_data/`, and creates `wood_data/manifest.csv`. The other notebooks work from these exported files rather than reopening the archive.

## Installation

The project used Python 3.12.13.

```bash
pip install -r requirements.txt
```

Keras runs on the PyTorch backend. A GPU is recommended for training; the final runs used mixed precision.

## Running the project

Download the source archive and place it at `hd5_dataset/WIG_v1.2.1_600.h5`.

Run the notebooks in numerical order:

1. `1. data_pipeline.ipynb` creates the splits, PNG dataset, and manifest.
2. `2. train_model.ipynb` trains and evaluates the frozen and fine-tuned models.
3. `3. feature_interpretation.ipynb` analyzes features from the fine-tuned model.

The notebooks depend on outputs from the previous stage. Reproducing the results requires the full XDD_015 archive and a complete run of all three notebooks.

## Sample data

`sample_data/` shows the image layout, manifest format, and loading code. It is too small to train the model or reproduce the reported results. See `sample_data/README.md` for details.

## References

1. Sugiyama, J., Hwang, S. W., Zhai, S., Kobayashi, K., Kanai, I., & Kanai, K. (2020). *Xylarium Digital Database for Wood Information Science and Education (XDD_015)* [Data set]. Research Institute for Sustainable Humanosphere, Kyoto University. [https://doi.org/10.14989/XDD_015](https://doi.org/10.14989/XDD_015)
2. Tan, M., & Le, Q. V. (2021). EfficientNetV2: Smaller models and faster training. In M. Meila & T. Zhang (Eds.), *Proceedings of the 38th International Conference on Machine Learning* (Vol. 139, pp. 10096-10106). PMLR. [https://proceedings.mlr.press/v139/tan21a.html](https://proceedings.mlr.press/v139/tan21a.html)
3. Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., & Li, F.-F. (2009). ImageNet: A large-scale hierarchical image database. In *2009 IEEE Conference on Computer Vision and Pattern Recognition* (pp. 248-255). IEEE. [https://doi.org/10.1109/CVPR.2009.5206848](https://doi.org/10.1109/CVPR.2009.5206848)
4. Wheeler, E. A., Baas, P., & Gasson, P. E. (Eds.). (1989). IAWA list of microscopic features for hardwood identification. *IAWA Bulletin n.s., 10*(3), 219-332.

