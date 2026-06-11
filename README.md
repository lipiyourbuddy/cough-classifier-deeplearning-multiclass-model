# Multi-Class Cough Classifier
A new method based on convolutional neural networks and deep feature extraction using pitch-shifting data augmentation for covid-19, asthma, other respiratory diseases and healthy.

We used COUGHVID and Coswara datasets.

- extract_asthma.py and extract_other_resp.py --> Used to extract asthma and other remaining audios from the coswara dataset using its csv file.

- FINAL_AUDIO folder --> contains the final segregated audios from both datasets before pre-processing and augmentation.

- preprocess_audio.py --> to preprocess audios files, including stripping silences, padding, splitting and chunking.

- augment_all.py --> augmentation includes adding noise, time-stretch, pitch-shift, time-shift and random gain.

- generate_spectrograms.py --> generates log-mel spectrograms with y-axis as frequency, x-axis as the time and colour grading as intensity. 

- train_Model--> Training the proposed methods

- test_Model.py--> Test phase

- evaluation.py--> plotROCCurve, plotConfusionMatrix, and metrics


Celik. G (2023). CovidCoughNet: A new method based on convolutional neural networks and deep feature extraction using pitch-shifting data augmentation for covid-19 detection from cough, breath, and voice signals. Computers in Biology and Medicine. 163, 107153.

doi: https://doi.org/10.1016/j.compbiomed.2023.107153.


