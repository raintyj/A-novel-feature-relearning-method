# A-novel-feature-relearning-method

Prepare dataset:
    We evaluated our model with the Sleep-EDF dataset. The data set is publicly available on the Internet http://www.physionet.org/physiobank/database/sleep-edfx/. When using this resource, please cite the original publication: B Kemp, AH Zwinderman, B Tuk, HAC Kamphuisen, JJL Oberyé. Analysis of a sleep-dependent neuronal feedback loop: the slow-wave micro continuity of the EEG. IEEE-BME 47(9):1185-1194 (2000). To comply with the AASM recommendations, this paper combines N3 and N4 stages into a single N3 and removes the movement and unknown data.

Environment:
   The following setup has been used to reproduce this work:
        Windows 10 64bit
        CUDA toolkit 10.0 and CuDNN v7.6.5
        Python (3.7.9)
        tensorflow-gpu (2.0.0)
        matplotlib (3.3.2)
        scikit-learn (0.23.2)
        scipy (1.5.2)
        numpy (1.19.4)
        pandas (1.1.5)
        imbalanced-learn (0.7.0)
