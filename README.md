# Identifying-cardiac-drug-effects-using-StreaMRAK

### Generate data
 - Run main_generate_data.py to generate the beeler-reuter AP traces over a domain

### Train models
 - Run main_train_models.py to the train StreaMRAK and FALKON models

### Make predictions using the models
- Run main_domain_predictions.py to run estimate parameters of AP traces (Drug effects) using the trained 
  StreaMRAK and FALKON models. This script also runs estimates using the euclidean and action potential features
  loss function minimization schemes.

### Visualization
Run main_visualize_domainpred.py

### Note
 - It might be necessary to comment-out matplotlib.use('TkAgg')
 - install packages using requirements.txt
