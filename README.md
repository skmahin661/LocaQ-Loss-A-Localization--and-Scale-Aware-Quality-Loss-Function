# LocaQ Loss

LocaQ Loss is a classification objective for medical abnormality detection. It extends Varifocal Loss (VFL) with adaptive foreground weighting based on classification confidence, localization quality, and target-box area. The YOLO integration replaces the classification-loss term during training and retains the detector architecture, box regression loss, and Distribution Focal Loss.

## Use LocaQ on your dataset

1. Clone or download this repository and launch Jupyter from its root so Python can import [locaq_loss.py](locaq_loss.py).
2. Select a kernel with PyTorch and Ultralytics. The original thyroid notebook records Python 3.10.20, PyTorch 2.5.1+cu121, and Ultralytics 8.4.51. The one-epoch thyroid smoke test completed with Python 3.11.17, PyTorch 2.11.0+cu128, and Ultralytics 8.4.51. Choose a PyTorch build for your hardware.
3. Open [the usage notebook](use_locaq_loss_with_yolo.ipynb), set `DATA_YAML`, choose YOLOv8m or YOLO11m with `MODEL_CFG`, and run the one-epoch example. It selects GPU 0 when CUDA is available and CPU otherwise.
4. Run the optional test-set evaluation if your dataset YAML has a `test` split.

A detection YAML can look like this. Point `path` to your dataset root and replace the example class names with the names in your labels.

```yaml
path: /absolute/path/to/dataset
train: images/train
val: images/val
test: images/test  # Optional
names:
  0: finding_a
  1: finding_b
```

The training callback constructs the loss class as follows:

```python
from locaq_loss import LocaQLoss
```

`LocaQLoss` is the implementation class.

## Default loss parameters

| Parameter | Default | Purpose |
| --- | ---: | --- |
| `beta` | 1.0 | Strength of the confidence- and localization-dependent multiplier |
| `vfl_alpha` | 0.75 | VFL background weighting coefficient |
| `vfl_gamma` | 2.0 | VFL background focusing exponent |
| `anchor_area` | 1024.0 | Reference area for target-box scale weighting |
| `fn_clamp_max` | 3.0 | Upper bound on the combined foreground multiplier |

The adaptive multiplier is computed without gradient tracking and applied when a batch contains more than 10 foreground assignments. Gradients still flow through the classification objective. For direct use of the helper outside the training callback, supply `xyxy` boxes in a common coordinate system for the overlap calculation and use consistent area units for the target boxes and reference area.

## Reproduce the YOLOv8m experiments

| Dataset | Notebook | Reported mAP50 | Reported mAP50–95 |
| --- | --- | ---: | ---: |
| Thyroid Histopathology | [Training and evaluation](thyroid_histopathology_yolov8m_locaq_loss.ipynb) | 0.6390 | 0.4558 |
| Dental OPG | [Training and evaluation](dental_opg_yolov8m_locaq_loss.ipynb) | 0.5963 | 0.3367 |
| Brain Tumor | [Training and evaluation](brain_tumor_yolov8m_locaq_loss.ipynb) | 0.7542 | 0.5460 |

Supply the corresponding dataset and annotation splits, then configure the dataset YAML, output directory, and optional checkpoint before running an experiment notebook. The paper reports training/validation/test counts of 327/45/99 for Thyroid Histopathology, 332/44/106 for Dental OPG, and 770/110/221 for Brain Tumor. The notebooks do not include the datasets. Matching split counts alone does not reproduce the original split membership.

The logged one-epoch thyroid run verifies the integration on that setup. The table reports the paper experiments, which are separate from that smoke test.

## License

The code is distributed under the [GNU Affero General Public License v3.0](LICENSE).

## Manuscript

*LocaQ Loss: A Localization- and Scale-Aware Quality Loss for Medical Abnormality Detection*.
