# LocaQ Loss

LocaQ Loss is a classification objective for object detection. It extends Varifocal Loss with adaptive foreground weights based on classification confidence, localization quality, and target-box area. The integration preserves the detector architecture, box regression loss, and Distribution Focal Loss.

## Install

Install the PyTorch build appropriate for your hardware, then run:

```bash
git clone https://github.com/skmahin661/LocaQ-Loss-A-Localization--and-Scale-Aware-Quality-Loss-Function.git
cd LocaQ-Loss-A-Localization--and-Scale-Aware-Quality-Loss-Function
python -m pip install "ultralytics==8.4.51" jupyterlab
python -m jupyterlab
```

## Prepare your dataset

Use any object detection dataset with bounding-box labels in Ultralytics YOLO format. Each image needs a matching label file for its annotations. Each label row contains `class_id x_center y_center width height`, with zero-based class IDs and coordinates normalized to the image dimensions. Put the labels in the `labels/` directory corresponding to each `images/` split.

```yaml
path: /absolute/path/to/your/dataset
train: images/train
val: images/val
test: images/test  # Optional
names:
  0: class_a
  1: class_b
```

## Choose your model and train

Open the [usage notebook](use_locaq_loss_with_yolo.ipynb) from the repository root. Set `DATA_YAML` to your dataset YAML and `MODEL_CFG` to the configuration of your detector. LocaQ can be used with any Ultralytics YOLO detection model that supports the training-loss interface used by `LocaQLoss`. The paper evaluated YOLOv8m and YOLO11m. Compatibility with other model versions depends on their training interface.

Run the notebook with `EPOCHS = 1` for a quick integration check. It selects GPU 0 when CUDA is available and CPU otherwise. The callback prints `LocaQ Loss configured for the training model.` when it installs the custom criterion. `model.trainer.best` gives the path to `best.pt` after training. Run the optional evaluation cell if your YAML defines a test split.

## Integrate LocaQ into your own training script

```python
import types
from ultralytics import YOLO
from locaq_loss import LocaQLoss

def install_locaq(trainer):
    trainer.model.init_criterion = types.MethodType(
        lambda model: LocaQLoss(model), trainer.model
    )

model = YOLO(MODEL_CFG)
model.add_callback("on_train_start", install_locaq)
model.train(data=DATA_YAML, epochs=1)
```

Define `MODEL_CFG` and `DATA_YAML` as paths to your model configuration and dataset YAML before running the example.

## Default loss parameters

| Parameter | Default | Purpose |
| --- | ---: | --- |
| `beta` | 1.0 | Strength of the confidence- and localization-dependent multiplier |
| `vfl_alpha` | 0.75 | VFL background weighting coefficient |
| `vfl_gamma` | 2.0 | VFL background focusing exponent |
| `anchor_area` | 1024.0 | Reference area for target-box scale weighting |
| `fn_clamp_max` | 3.0 | Upper bound on the combined foreground multiplier |

The adaptive multiplier is computed without gradient tracking and applied when a batch contains more than 10 foreground assignments. Gradients still flow through the classification objective. For direct use of the helper outside the training callback, supply `xyxy` boxes in a common coordinate system for the overlap calculation and use consistent area units for the target boxes and reference area.



## Paper experiment notebooks

- [Thyroid Histopathology](thyroid_histopathology_yolov8m_locaq_loss.ipynb)
- [Dental OPG](dental_opg_yolov8m_locaq_loss.ipynb)
- [Brain Tumor](brain_tumor_yolov8m_locaq_loss.ipynb)

A one-epoch thyroid run completed with Ultralytics 8.4.51, Python 3.11.17, and PyTorch 2.11.0+cu128. See the experiment notebooks for the paper configurations.

## Reported paper results

| Dataset | YOLOv8m mAP50 | YOLOv8m mAP50–95 | YOLO11m mAP50 | YOLO11m mAP50–95 | YOLO11m mAP50–95 gain over BCE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Thyroid Histopathology | 0.6390 | 0.4558 | 0.6417 | 0.4917 | +2.23 pp |
| Dental OPG | 0.5963 | 0.3367 | 0.6384 | 0.3570 | +5.71 pp |
| Brain Tumor | 0.7542 | 0.5460 | 0.7565 | 0.5147 | +0.55 pp |

The table shows the reported LocaQ Loss scores. **pp** means percentage points. With YOLOv8m, LocaQ achieved the highest mAP50 and mAP50–95 among the five evaluated losses on all three datasets. The largest mAP50–95 gain over the strongest YOLOv8m baseline was 3.79 pp on Thyroid Histopathology.

With YOLO11m, LocaQ improved mAP50–95 over default BCE on all three datasets, with the largest gain on Dental OPG. On Brain Tumor, mAP50–95 improved while mAP50 decreased from 0.7787 to 0.7565. The reported comparisons used a single training seed.

## License

[GNU Affero General Public License v3.0](LICENSE)
