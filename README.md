# Autodamage
### Classification of cars and car damage to estimate costs of repairs
### By Christian Meraz-Santiago
###### Note: I do not include the data folder in the repo, all data links are provided and can be downloaded from the original source.
###### Model checkpoints are not included as they are reproducible.
The end goal of this project is to have a deployable where a user can provide an image of a damage on a vehicle and get a prediction of the cost of repairs and be provided with general information about what is damaged, information about the particular vehicle, and locations to get it fixed nearby. Many mechanics/body-shops are infamously known to upcharge unknowing customers, so being able to easily get an estimate of what the repairs should cost along with helpful information should give the user a good idea of what they are in for.

### Planned Pipeline

#### 1. Car Classification
Fine-tune a vision model on labeled images.

#### 2. Damage detection/severity
Fine-tune a model (thinking possibly Gemma 4) to detect what is damaged and how severe - bumper dented vs. crumpled, scratched vs.torn, etc.
Be able to output label as (part + severity).

#### 3. Cost Prediction
Feed outputs from steps 1 and 2 plus any additional features ( such as labor rates, parts cost data) into a model (Maybe XGBoost/LightGBM) or just a regression network.

#### 4. Text Report Generation
Once I have structured data (Car ID + damage list + cost estimate), use an LLM to turn that data into a natural-language report.

#### 5. Repair Shop Recommendation
Use an API such as Google Places or Yelp filtered by location to find nearby body-shops/mechanics for actual repairs that would closely 

## Car Classification
### Dataset being used for car classification: 
#### https://www.kaggle.com/datasets/eduardo4jesus/stanford-cars-dataset

The first step for this project is car classification. 