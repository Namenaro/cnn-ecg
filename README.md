# cnn-ecg
The repository contains code for the paper "Problems of representation of electrocardiograms in convolutional neural networks"
 
1) gaussian_experiment - this is the code, reproducing experiments with "bad" and "good" networks; precisely, it reprodiceses reaction of "bad" and "good" convolutional networks in the ECG vs gaussian trajectories. First, we select some ECGs from the dataset. Second, we generate gaussian trajectories for these ECGs (i.e. with thw same means and std-s). Next, the bad and the good network are trained on ECGs and stochasic signals. 
2) interpolation_experiments contains the all code for ECG-interolation part of the paper: training, visualisation, interpolation
