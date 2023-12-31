# MFC
We propose a new video object
segmentation model that utilizes self-supervised learning to extract spa-
tial features, and incorporates a motion feature, extracted from optical
flow, as compensation of temporal information for the model, namely mo-
tion feature compensation (MFC) model. Additionally, we introduce an
attention-based fusion method to merge features from both modalities.
Notably, for each video used to train models, we only select two consec-
utive frames at random to train our model. The dataset Youtube-VOS
and DAVIS-2017 are adopted as the training dataset and the valida-
tion dataset. The experimental results demonstrate that our approach
outperforms previous methods, validating our proposed design.

![image](6FD57612-FA81-4088-8DA6-DB997D9DBEAA.jpeg)

# Requirements
The code has been trained and tested with PyTorch 1.9 (1.9.0a0+gitc91c4a0), Python 3.9, and Cuda 11.2.

Other dependencies could be installed by running:
```Shell
pip install -r requirements.txt
```

# Use  
Training：  
--------
sh train.sh  

Evaluation:  
--------
sh test_davis.sh  
