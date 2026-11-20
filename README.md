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

## Stage 2b (in progress): Part Detection + Repair Action
CarDD labels damage type but not *which part* is damaged, and the cost stage needs the part and the likely repair.
### Part detection
-Dataset: Ultralytics Carparts-seg (https://docs.ultralytics.com/datasets/segment/carparts-seg), 3,833 images, converted to COCO with left/right variants merged into 12 classes (front/back bumper, hood, trunk, front/back door, front/back light, front/back glass, mirror, wheel)  
-Model: Mask R-CNN, same training code as the damage detector (`python -m src.damagedetection.train --data-root data/carpartsseg --img-dir-pattern "images/{split}" --checkpoint models/part_detector/maskrcnn_carparts_best.pth --epochs 15`)  
-Known gaps: undamaged whole-car photos (CarDD is mostly close-ups), and no fender / quarter panel / roof / rocker classes. Planned fix: add the Humans in the Loop "Car Parts and Car Damages" dataset (CC0, damaged cars)  
### Repair action
No dataset labels severity, and the cost stage doesn't need an abstract severity anyway: it needs the repair. So each damage goes straight to an action, using the same names as the cost estimator:
-Each damage mask is matched to the part mask it overlaps most; **extent** = fraction of that part the damage covers  
-Glass shatter / lamp broken -> replace; flat tire -> repair (plug)  
-Crack or dent on glass or a light -> replace  
-Dent: extent < 5% -> paintless dent repair, < 20% -> repair and refinish, else replace  
-Scratch: < 25% -> refinish only, else repair and refinish (never replace)  
-Crack: < 10% -> repair and refinish, else replace  
-No part matched -> same idea using the fraction of the image  
Rules live at the top of `src/damagedetection/repair_action.py`. A sanity check against 100 hand severity labels (`labels/severity_labels.csv`; refinish/PDR ~ minor, repair ~ moderate, replace ~ severe): `python -m src.damagedetection.evaluate_repair_action --search`
### Results
**Part model** (15 epochs, best = epoch 8 by val loss), Carparts-seg test set: mask AP 0.671, AP50 0.884. Strong on bumpers, hood, doors, windshield (mask AP 0.72-0.84); weak on wheels (0.28, many unlabeled wheels in the dataset count as false positives).

**On CarDD damage photos** the picture is different:
-Only 46/100 hand-labeled damages land on a detected part; the rest fall back to image extent. CarDD is mostly close-ups, and the part model was trained on whole, undamaged cars  
-Matched parts were almost all hood/bumper; no doors were matched  
-On real accident photos the *damaged* parts themselves often go undetected (a crushed bumper doesn't look like the undamaged bumpers it learned), and damage near a boundary can be assigned to the wrong part (e.g. a hood dent assigned to the windshield)  

**Repair-action rules vs. 100 hand labels:** 56% level agreement (3 levels). Fixed-type rules are reliable (glass/lamp 100%, tire 80%); dent 40%, crack 45%, scratch 53%. Most misses under-call severe damage: the rules see only the outlined damage, while the repair depends on the state of the whole panel. A grid search reached 67% but with degenerate cutoffs (overfit to these labels), so the hand-set rules were kept.

**Next:** retrain the part model with damaged-car data (Humans in the Loop dataset: damaged cars, adds fender / quarter panel classes) so damaged parts are recognized; then revisit the rules.

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
