EEG_AI_COMPETITION SUBMISSION
Created by Zimo He and Yichen Xu
Contact us via michaelxu_cn@outlook.com

Requirement:

python 3.9.12
tensorflow 2.10.0
numpy 1.21.5
scipy 1.7.3

How to implement the code:

1.Download data_EEG_AI.mat and put it under the root directory

2.We've saved the trained model under 'model_data' directory. 
IF you want to see the final performance, run: python weighting.py
IF you want to train your own model, run: python TCN Training.py
then get the performance by running: python weighting.py

3.Use get_channel_weight.py to get channel weights based on PCA and KNN, you can also design your own channel weights.

4.The code of KNN and our PCA preprocessing method are provided. You may also want to check on it. 

Method:

1.Model Selection
We've tried different deeep learning methods to solve the 26-classification problem, including LSTM, ResNet, CNN and TCN. 
It turns out that a channel-weighted TCN network performs the best, achieving a 13.7% accuracy in our experiment.
We've reviewed many prior studies before starting our work and we discovered that the TCN network could be outstanding in performance 
for its ability to reveal the temporal information of the brain signal. The link of the article is attached below.

2.Channel-Level Weights
After our data-analysis, we found that different channel may not contribute to the task equally. 
Therefore, we decided to give each channel different weights determined by a KNN method.

3.Time Window
To discuss the effectiveness of different time steps, we visualized the raw data and found out that the brain signal is not always active
from the beginning to the end. However, the siganl is relatively active in the 0-1000ms time window. Based on this discovery, we trained our
model on different time windows and realized that 0-1000ms had better performance compared with others. 
We inferred that it might not take 3s for the one-character handwriting imagination process 
or the Signal-Noise ratio of beginning time steps is higher than the rest parts.

Reference: 

Multiclass Classification of Imagined Speech Vowels and Words of Electroencephalography Signals Using Deep Learning
Nrushingh Charan Mahapatra and Prachet Bhuyan
https://doi.org/10.1155/2022/1374880

Further Work:

1.We currently reviewed the time window roughly by 1000ms level, it may perform better if we divide the time steps into smaller time windows.

2.More preprocessing methods are yet to be applied to enhance the signal-noise ratio of the raw data.