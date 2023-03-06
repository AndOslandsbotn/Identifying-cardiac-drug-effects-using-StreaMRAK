# Identifying-cardiac-drug-effects-using-StreaMRAK

## Data
Data can be accessed in two ways.
#### From data file stored in git LFS
  - Download the git large file system (git LFS), then pull this repository and unzip the file BeelerReuter_idx[1, 3].zip.
#### Generate data
 - Run main_generate_data.py to generate the beeler-reuter AP traces over a domain. 
 - The generated data will be stored in the folder Data/BeelerReuter_idx[1, 3]

## Using StreaMRAK and FALKON
#### Train models
 - Run main_train_models.py to the train StreaMRAK and FALKON models

#### Make predictions using the models
- Run main_domain_predictions.py to estimate parameters of AP traces (Drug effects) using the trained 
  StreaMRAK and FALKON models. This script also runs estimates using the euclidean and action potential features
  loss function minimization schemes.

#### Visualization
 - Run main_visualize_domain_pred.py

## Notes
 - It might be necessary to comment-out matplotlib.use('TkAgg')
 - Install packages using requirements.txt
