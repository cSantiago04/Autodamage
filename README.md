# Autodamage
### Classification of cars and car damage to estimate costs of repairs
### By Christian Meraz-Santiago
###### Note: I do not include the data folder in the repo, all data links are provided and can be downloaded from the original source.
###### Model checkpoints are not included as they are reproducible.
The end goal of this project is to have a deployable where a user can provide an image of a damage on a vehicle and get a prediction of the cost of repairs and be provided with general information about what is damaged, information about the particular vehicle, and locations to get it fixed nearby. Many mechanics/body-shops are infamously known to upcharge unknowing customers, so being able to easily get an estimate of what the repairs should cost along with helpful information should give the user a good idea of what they are in for.

## Planned Pipeline

#### 1. Car Classification
Identify make/model/year. Baseline: fine-tuned ResNet-50 on Stanford Cars (done). Next: compare against Gemma 4 zero-shot, since Stanford Cars only covers 196 models up to 2012.

#### 2. Damage detection/severity
Mask R-CNN localizes damage and classifies its type (done). Next: use Gemma 4 (zero-shot first, LoRA fine-tune if needed) to label which part is damaged and how severe - bumper dented vs. crumpled, scratched vs. torn, etc.
Be able to output label as (part + severity).

#### 3. Cost Prediction
Feed outputs from steps 1 and 2 plus any additional features (such as labor rates, parts cost data) into a model (Maybe XGBoost/LightGBM) or just a regression network. A rules-based estimate (labor hours x regional rate + part price range) is the likely first version until real repair-cost data is found.

#### 4. Text Report Generation
Once I have structured data (Car ID + damage list + cost estimate), use an LLM to turn that data into a natural-language report.

#### 5. Repair Shop Recommendation
Use an API such as Google Places or Yelp filtered by location to find nearby body-shops/mechanics for actual repairs.

## Demo
A local web page for trying the pipeline on any photo: damage and parts drawn on the image, plus a table of damage -> part -> repair action (and the decoded vehicle if a VIN is entered).
```
python -m src.app            # then open http://127.0.0.1:7860
```

## Current Repository Structure
```
Autodamage/
├── data/                                  # Raw datasets (gitignored, not tracked)
│   ├── carclassificationdataset1/         # Stanford Cars
│   ├── damagedetectiondataset1/           # COCO Car Damage Detection (Scrapped)
|   └── CarDD_release                      # Huge dataset for damage detection
├── models/                                # Trained checkpoints (gitignored, not tracked)
│   ├── classifier/
│   └── damage_detector/
├── notebooks/                             # Exploration and prototyping
│   ├── carclassification.ipynb
│   └── damagedetection.ipynb
├── src/
│   ├── carclassification/
│   │   ├── data_loader.py                 # Stanford Cars parsing + DataLoaders
│   │   ├── train.py                       # ResNet-50 training loop
│   │   ├── evaluate.py                    # Test set evaluation + confusion matrix
│   │   ├── prediction.py                  # Single-image ResNet prediction
│   │   ├── vin_decoder.py                 # VIN -> year/make/model/trim via NHTSA vPIC
│   │   └── vehicle_detector.py            # COCO-pretrained Faster R-CNN: find + crop the damaged vehicle
│   ├── damagedetection/
│   │   ├── data_loader.py                 # Parses CarDD COCO annotations, builds DataLoaders
│   │   ├── train.py                       # Model definition + training loop, saves best checkpoint
│   │   ├── evaluate.py                    # Test set evaluation
│   │   ├── evaluate_map.py                # COCO mAP (box + mask), per class, precision/recall at a cutoff
│   │   ├── prediction.py                  # Single-image damage prediction (boxes, optional masks)
│   │   ├── convert_carparts.py            # Carparts-seg YOLO polygons -> COCO, 12 merged part classes
│   │   ├── part_prediction.py             # Single-image car-part segmentation
│   │   ├── repair_action.py               # Damage -> part matching (mask overlap) + repair-action rules
│   │   ├── label_severity.py              # Labeling tool: hand-label CarDD damage severity
│   │   └── evaluate_repair_action.py      # Rules vs. hand severity labels (+ cutoff grid search)
│   ├── gemma/
│   │   ├── client.py                      # Ollama client for Gemma 4 (image + JSON-schema output)
│   │   ├── eval_car_classification.py     # Zero-shot Gemma vs. ResNet on the Stanford Cars test split
│   │   ├── identify_car.py                # Open-set make/model/trim/year from a vehicle crop
│   │   └── describe_damage.py             # Mask R-CNN + Gemma: vehicle, damaged part, severity (+ --estimate)
│   ├── costestimation/
│   │   ├── cost_tables.json               # Editable labor hours, rates, part prices (rough assumptions)
│   │   └── estimator.py                   # Rules-based damage list -> repair cost range
│   ├── app.py                             # Gradio demo page (local)
│   └── pipeline.py                        # VIN + damage -> vehicle -> parts -> repair action
├── tests/
│   ├── test_estimator.py                  # pytest tests for the cost estimator
│   ├── test_vin_decoder.py                # pytest tests for VIN validation/decoding
│   └── test_repair_action.py              # pytest tests for part matching + repair-action rules
├── labels/
│   └── severity_labels.csv                # Hand severity labels for CarDD test damages
├── .gitignore
├── README.md
└── requirements.txt
```

## Stage 1: Car Classification
Classifies the make/model/year of a car from an image using a ResNet-50 model.
### Dataset being used for car classification: 
https://www.kaggle.com/datasets/eduardo4jesus/stanford-cars-dataset

### Approach
-Images are cropped by their bounding box before resizing  
-Split: 70% Train, 15% val, 15% test, stratified by class for balanced representations across all 196 classes  
-Model: ResNet-50 pretrained on ImageNet, fine-tuned with swapped final layer for 196 classes  
-Data Augmentation (Random horizontal flip, rotation, color jitter) Applied to training set  
-Best checkpoint saved based on validation accuracy, not just the final epoch

### Results
No augmentation, 10 epochs |79.62% test accuracy|  
(Best checkpoint) With augmentation, 20 epochs |84.37% test accuracy|

### Error Analysis
Confusion matrix shows misclassifications are concentrated on similar versions of the same car, such as different body styles (coupe vs. convertible), trims, or adjacent model years.

## Stage 2: Damage Detection
Detects and localizes vehicle damage from an image using a Mask R-CNN model, identifying bounding box and damage type (scratch, dent, etc.)
### Dataset being used for damage detection:
https://drive.google.com/file/d/1bbyqVCKZX5Ur5Zg-uKj0jD0maWAVeOLx/view  
Citation:  
@article{wang2023cardd,  
  title={CarDD: A New Dataset for Vision-Based Car Damage Detection},  
  author={Wang, Xinkuang and Li, Wenjing and Wu, Zhongcheng},  
  journal={IEEE Transactions on Intelligent Transportation Systems},  
  volume={24},  
  number={7},  
  pages={7202--7214},  
  year={2023},  
  doi={10.1109/TITS.2023.3258480}
}
###### Early prototyping done on this dataset: (https://www.kaggle.com/datasets/lplenka/coco-car-damage-detection-dataset)

### Approach
-COCO-format annotations parsed into Pytorch's detection format  
-Mask R-CNN with ResNet-50 FPN backbone, fine-tuned for 6 damage classes (+ background)
-Data Augmentation applied  
-Trained with SGD + momentum, using a step learning rate scheduler  
-Best checkpoint saved based on validation loss  
-Non-maximum suppression (NMS) threshold tuned (0.3) to reduce overlapping box predictions  

### Results

CarDD, 10 epochs, fixed LR, Val_Loss = 0.6055 | Test_Loss = 0.5758  
(Best Checkpoint) CarDD, 15 epochs, LR scheduler, Val_loss = 0.5772 | Test_Loss  = 0.5450

### COCO Evaluation (CarDD test set, 374 images)
`python -m src.damagedetection.evaluate_map`

| Class | Box AP | Mask AP | Mask AP50 | Precision @0.5 conf | Recall @0.5 conf |
|---|---|---|---|---|---|
| dent | 0.300 | 0.317 | 0.632 | 0.50 | 0.73 |
| scratch | 0.302 | 0.275 | 0.618 | 0.49 | 0.73 |
| crack | 0.221 | 0.121 | 0.341 | 0.42 | 0.63 |
| glass shatter | 0.822 | 0.852 | 0.981 | 0.87 | 0.94 |
| lamp broken | 0.555 | 0.597 | 0.754 | 0.55 | 0.86 |
| tire flat | 0.795 | 0.828 | 0.904 | 0.64 | 0.94 |
| **All** | **0.499** | **0.498** | **0.705** | | |

Glass shatter and flat tires are reliable; dents, scratches and especially cracks are the weak classes. At the app's 0.5 confidence cutoff about half of dent/scratch detections don't match a labeled damage (some of that is ambiguous extents, e.g. one long scratch predicted as two). Raising the cutoff to 0.8 lifts dent/scratch precision to ~0.70 at ~0.55 recall.

### Error Analysis
-Occasional damage-type confusion on similar categories, such scratches near headlights being misclassified as "lamp broken"  
-False-postive detections on undamaged areas (especially true for flat tire classifications)  
-Further improvement and tuning is planned

## Stage 2b: Part Detection + Repair Action
CarDD labels damage type but not *which part* is damaged, and the cost stage needs the part and the likely repair.
### Part detection
-**Stage 1:** Mask R-CNN trained on Ultralytics Carparts-seg (https://docs.ultralytics.com/datasets/segment/carparts-seg), 3,833 mostly undamaged whole-car photos, 12 merged part classes  
-**Stage 2 (current model):** fine-tuned from that checkpoint on the Humans in the Loop "Car Parts and Car Damages" dataset (https://humansintheloop.org/resources/datasets/car-parts-and-car-damages-dataset/, CC0): 998 photos of damaged cars, 18 part classes including fender, quarter panel, rocker panel, roof, grille  
-Note: in that download the folder names are swapped ("Car damages dataset" holds the part labels). 441 photos have both part and damage labels; 120 of them are held out as the test set  
-Fine-tuning (not pooling the datasets) because Carparts-seg never labels fenders / quarter panels / side windows, so pooled training would call those "background" on half the photos  
-Overlapping masks of different parts are de-duplicated (one panel, one name)  
```
python -m src.damagedetection.convert_hitl
python -m src.damagedetection.train --data-root "data/Car parts and car damages dataset" --img-dir-pattern "Car damages dataset/File1/img" --checkpoint models/part_detector/maskrcnn_hitl_finetuned_best.pth --init-from models/part_detector/maskrcnn_carparts_best.pth --epochs 20
python -m src.damagedetection.evaluate_part_matching --checkpoint models/part_detector/maskrcnn_hitl_finetuned_best.pth
```
### Repair action
No dataset labels severity, and the cost stage doesn't need an abstract severity anyway: it needs the repair. So each damage goes straight to an action, using the same names as the cost estimator:
-Each damage mask is matched to the part mask it overlaps most; **extent** = fraction of that part the damage covers  
-Glass shatter / lamp broken -> replace; flat tire -> repair (plug)  
-Crack or dent on glass, a light, or the grille -> replace  
-Dent: extent < 5% -> paintless dent repair, < 20% -> repair and refinish, else replace  
-Scratch: < 25% -> refinish only, else repair and refinish  
-Crack: < 10% -> repair and refinish, else replace  
-No part matched -> same idea using the fraction of the image  
-**Collision context:** if the photo shows a broken lamp / shattered glass or 4+ separate damages, dent/scratch/crack actions go up one level. On the hand labels the outlined damage's own size barely predicts the repair (severe damages had a *smaller* median outline than moderate ones); a small dent on a panel that took a real hit still means replacement  

Rules live at the top of `src/damagedetection/repair_action.py`. Sanity check against 100 hand severity labels (`labels/severity_labels.csv`, labeled by Claude from the images; refinish/PDR ~ minor, repair ~ moderate, replace ~ severe): `python -m src.damagedetection.evaluate_repair_action`
### Results
**Is each damage matched to the right part?** 1,413 labeled damages on the 120 held-out HITL photos (ground-truth damage masks):

| Part model | Correct part | Wrong part | No part found |
|---|---|---|---|
| Carparts-seg only | 49.8% | 24.6% | 25.6% |
| HITL, trained from COCO weights | 82.4% | 15.3% | 2.3% |
| **Carparts-seg -> fine-tuned on HITL** | **88.0%** (87.5% with de-duplication) | 9.8% | 2.3% |

Per part (fine-tuned): front bumper 94%, front door 92%, back bumper 92%, hood 90%, quarter panel 85%, fender 83% (0% before: no class), headlight 82%; weakest is back door 63% (confused with front door / quarter panel). Fine-tuning mostly helps doors: training from COCO weights confused front and back doors 106 times.

**On CarDD close-ups:** damages landing on a detected part went from 46/100 to 84/100.

**Repair actions vs. 100 hand labels:** 56% with extent rules alone -> 69% with collision context -> **71%** with the fine-tuned part model (crack 80%, dent 60%, scratch 60%, lamp / glass 100%, tire 80%).

**Remaining weak spot: the damage detector.** On real photos it still produces false positives on intact glass (a "dent" covering an undamaged windshield) and oversized masks, and misses some creases. Next: retrain/extend it (the HITL damage half adds "broken part" / "missing part" labels, direct replace signals CarDD lacks).

## Gemma 4 Evaluation (in progress)
Testing whether a general vision-language model (Gemma 4, run locally via Ollama) can replace or complement the custom models, especially for part + severity labels that CarDD doesn't provide.
### Setup
1. Install Ollama: https://ollama.com/download
2. `ollama pull gemma4:e4b` (fits a 12GB GPU; try `gemma4:12b` for comparison)
### Run (from repo root)
```
python -m src.gemma.eval_car_classification --mode closed --limit 200
python -m src.gemma.eval_car_classification --mode open --limit 200
python -m src.gemma.describe_damage "data/Test Images"
```
### Results
Zero-shot gemma4:e4b on 200 images of the same Stanford Cars test split:

| Mode | Metric | Gemma 4 e4b | ResNet-50 (fine-tuned) |
|---|---|---|---|
| Closed (pick 1 of 196 classes) | exact class | 31.0% | 84.37% |
| Open (name it freely) | make | 61.5% | - |
| Open | make + model | 19.0% | - |
| Open | year within 1 | 21.0% | - |

On 3 real damaged-car photos (Civic Si coupe, Altima coupe, Acura RSX, none of which are in Stanford Cars) the ResNet got 0/3; Gemma and Qwen2.5-VL got 3/3 makes but only 1/3 models.
**Decision:** identifying the exact car from a photo isn't reliable enough for pricing, so the app gets the vehicle from its **VIN** (decoded with NHTSA's free vPIC API, `src/carclassification/vin_decoder.py`). A vision model can still pre-fill a guess for the user to confirm.

## Stage 3 (in progress): Cost Estimation
Turns the damage list (part + damage type + severity + likely repair) into a low/high repair cost range.
### Approach
-No public dataset of labeled repair costs exists, so v1 is rules-based, structured like a real body-shop estimate: labor hours x labor rate + parts + paint materials (billed per refinish hour), with flat prices for paintless dent repair, windshield chips, tire plugs, etc.  
-Free-text part names are mapped to ~18 canonical parts (e.g. "driver front door" -> door, front-left)  
-Multiple damages on one panel are merged into one repair at the worst severity (the panel is painted once)  
-Repair action comes from Gemma's `likely_repair` when given, otherwise a default by damage type + severity  
-Luxury/exotic makes scale part and labor costs  
-Every number is a [low, high] range, so the output is always a range  
### Caveat
The numbers in `src/costestimation/cost_tables.json` are rough ballpark assumptions, **not sourced data**. They need calibrating against real repair quotes. Edit the JSON to change them; no code changes needed.
### Run
```
python -m src.costestimation.estimator                              # demo estimate
python -m src.gemma.describe_damage "data/Test Images" --estimate   # photo -> damage -> cost
python -m pytest tests/
```
