# Identifying-cardiac-drug-effects-using-StreaMRAK

## Data
Data can be accessed in three ways.
#### Alternative 1: From data file stored in git LFS
  - Alternative 1a: Download the git large file system (git LFS), then pull this repository and unzip the file BeelerReuterData.zip. This will give a files BeelerReuter_idx[1, 3] which should be placed in the folder Data. This folder will contain:
    - DomainData_N10000.npz,
    - Training data TrData_NX.npz for X=[100, ..., 6000] 
    - Test data TsData.npz
  - Alternative 1b: Pull the data in Data/DomainData_N10000.7z. Unzip the data to get DomainData_N10000.npz. This data should be placed in a folder Data/BeelerReuter_idx[1, 3]. 
    Then run the main_generate_data.py only with TestTrainDataGenerator (with DataGenerator commented out). This generates the test and training data
    - Training data TrData_NX.npz for X=[100, ..., 6000] 
    - Test data TsData.npz
#### Alternative 2: Generate data
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
