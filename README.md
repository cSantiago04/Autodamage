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
│   │   └── evaluate.py                    # Test set evaluation + confusion matrix
│   └── damagedetection/
│       ├── data_loader.py # Parses CarDD COCO annotations, builds DataLoaders
│       ├── train.py # Model definition + training loop, saves best checkpoint
│       └── evaluate.py # Test set evaluation
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

### Error Analysis
-Occasional damage-type confusion on similar categories, such scratches near headlights being misclassified as "lamp broken"  
-False-postive detections on undamaged areas (especially true for flat tire classifications)  
-Further improvement and tuning is planned