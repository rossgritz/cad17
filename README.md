# cad17
### Source code for the paper: 3d Deep Learning for Detecting Pulmonary Nodules in CT Scans
### Ross Gruetzemacher, Ashish Gupta, David Paradice
### The Journal of the American Medical Informatics Association
## Instructions for use:
### 1. Download the datasets:
###    - LUNA16 dataset (https://luna16.grand-challenge.org/)
###    - LIDC-IDRI dataset (https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI)
### 2. Generate candidates
###    - Generate images and masks for training using segutil.py
###       - Enter filepaths in source code and execute 
###    - Train U-NET DNN with seg0.py
###       - Enter filepaths in source code and execute 
###    - Generate candidates with candgen.py
###       - Enter filepaths in source code and execute 
###    - Evaluate segmentation with segment*.py
###       - Enter test bin, validation bin and model 
### 3. Reduce false positives
###    - Remove noise from small candidates with fpr.py
###    - Select parameters in config.json
###    - Train false positive reduction network with: keras/generatorB.py
###    - Remove ambiguous nodules with util/ambiguous.py
###    - Evaluate false positive reduction with eval.py
## Notes
### The files here were used as described in the Appendix of the paper. Use on other systems may require extensively modifying the scripts and source code. Scripting code can be found in the /utils folder.
### The solutions can be found at: https://www.dropbox.com/sh/mzw9y9ktl0ngmjq/AACC4WKvIY76RaCHNZKrhO6va?dl=0
