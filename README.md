# cad17
## Instructions for use:
### 1. Download the datasets:
###    - LUNA16 dataset (https://luna16.grand-challenge.org/)
###    - LIDC-IDRI dataset (https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI)
### 2. Generate candidates
###    - Generate images and masks for training using segutil.py
###       - Enter filepaths in source code and execute 
###    - Train U-NET DNN with seg2.py
###       - Enter filepaths in source code and execute 
###    - Generate candidates with candgen.py
###       - Enter filepaths in source code and execute 
### 3. Reduce false positives
###    - Remove noise from small candidates with fpr.py
###    - Select parameters in config.json
###    - Train false positive reduction network with: keras/generatorModel.py
###    - Remove ambiguous nodules with util/ambiguous.py
###    - Evaluate false positive reduction with notebooks/Evaluate.ipynb
