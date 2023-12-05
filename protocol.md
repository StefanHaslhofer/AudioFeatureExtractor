# Exercise PC Design & Development / Systems and Environments

## Assignment 1



**11908757 Stefan Haslhofer**

##### 1. Remarks

My dataset used for the vehicle type classification may share similarities with the dataset of Jonas Reichhardt as we were meeting up and recording at the same location. Nonetheless, I did the assignment completely on my own. 

Due to cold weather, I had to reduce the vehicle types to only medium and heavy as there has not been a single 'light vehicle' (motorbike) in sight.

There is a heavy bias on cars which seems reasonable as most registered motorized vehicles are indeed cars. Furthermore, I recorded the first dataset (used for the vehicle type classification) during the evening rush hour, which meant nearly all vehicles were heading out of town. Unfortunately, this rendered the first dataset useless for the driving direction classification. Hence, I created a second less biased dataset.

Concerning the raw audio we should include in our submission: I merged all files to one big *wav-file* per classification task. Therefore, I will hand in the **whole raw dataset** alongside all labels with start- and end timestamps as well as all feature sets (according to https://moodle.jku.at/jku/mod/forum/discuss.php?d=118760).

Python code: https://github.com/StefanHaslhofer/AudioFeatureExtractor

**Recording location of vehicle type classification dataset:**

![image-20231204210530990](C:\Users\haslh\AppData\Roaming\Typora\typora-user-images\image-20231204210530990.png)

**Recording location of the driving direction classification dataset:**

![image-20231205190823776](C:\Users\haslh\AppData\Roaming\Typora\typora-user-images\image-20231205190823776.png)

I recorded multiple short videos of the traffic with my mobile phone at approximately the distances shown in the sketch for the sample recording in the assignment sheet.

##### 2. Pre-processing

At first, I merged the raw video data into one large *mp4*-file for each classification task which I converted to *wav* and then labelled within the audio editing software *Audacity*. Due to the recording location having strong background noise further pre-processing was needed. However, *Audacity* provides a noise removal tool which I used to reduce background noise by 6 dB.

Next, I switched to a spectrogram view making it much easier to distinguish passing vehicles from remaining noise as some frequencies  suddenly spike in amplitude. Furthermore, I used the video as visual evidence to help me determine the correct driving direction for each sample. Labels can be exported from *Audacity* with start and end timestamp.

![image-20231204220733870](C:\Users\haslh\AppData\Roaming\Typora\typora-user-images\image-20231204220733870.png)

| start      | end        | vehicle_type | direction |
| ---------- | ---------- | ------------ | --------- |
| 137.657700 | 140.323168 | medium       | LR        |
| 142.714615 | 146.575807 | medium       | LR        |

**Note:** After exporting the labels to a *csv*-File I renamed *Car* to *medium* and *Truck*, *Bus* to *heavy*. 

##### 3. Feature extraction

Next, I implemented a python script slicing the large audio into multiple smaller pieces, one for each label (start and duration indicated by the timestamps). For each audio slice I extracted 13 features based on Mel-frequency cepstral coefficients (MFCCs). I used  the *librosa*-library's MFCC implementation which breaks the audio into windows and per default calculates 20 coefficients for each window. The chosen window size is equivalent to the number of coefficients in milliseconds.

However, I found that the overall number of correctly classified samples peaks at around 40 MFCCs for the used classifiers.

The following table lists all extracted features and why I initially chose them:

| feature             | remark                                                       |
| ------------------- | ------------------------------------------------------------ |
| mean                | I suspected that there must be a correlation between vehicle size/distance to the vehicle and the mean energy for each MFCC. |
| mean squared        | I added the squared mean to emphasize the larger overall sound energy of more powerful vehicles. |
| standard deviation  | Looking at the spectrogram, one may notice that heavier vehicles have a strong representation in lower frequencies, whereas lighter vehicles tend to be more spread out over the whole spectrum which means there could be a noticeable difference in the deviation from the mean. |
| variance            | I also included variance for the same reason as standard deviation. |
| median              | Similar to mean, the larger and closer a vehicle is the higher the amplitudes will be for each frequency, therefore the median energy should also be larger. |
| max energy          | I thought the dominant amplitude of heavier/closer vehicles stands out compared to the dominant amplitude of lighter/distant vehicles. |
| max energy bin      | I wanted to compare the dominant MFCCs because the maximal energy on its own may be misleading. For example: the amplitude of the loudest frequency (*max energy*) of a truck and a motorbike could be roughly the same even tough the motorbike's dominant frequencies are higher. |
| min energy          | Similar to the *max energy* feature.                         |
| min energy bin      | Related to *max energy bin* feature: the least dominant MFCC of two vehicle classes may differ even tough the largest amplitude does not. |
| q1                  | The frequencies do not seem to be distributed the same for all classes according to the spectrogram. Hence, the quartiles might be different. |
| q3                  | see *q1*                                                     |
| skewness            | After studying the spectrogram, I noticed the frequency distribution of heavier vehicles is possibly skewed to lower frequencies. Frequencies of lighter vehicles on the other hand seem to be more symmetrically arranged. |
| interquartile range | If frequency distribution differs widely between classes, so does the interquartile range. |

In general I assumed that features that provide information about frequency distribution are more useful to determine the vehicle type whereas features that hold information about the loudness (energy/amplitude) are a good indication for distance measuring and therefore direction (vehicles going from left to right are closer and should therefore be louder).

 ##### 4. Classification of vehicle type

*Weka* was used for classifications.

After evaluating the features I found that all vehicle types tend to be silent in a similar frequency range and also have akin maximal amplitudes, rendering *max energy*, *max energy bin*, *min energy* and *min energy bin* useless. Therefore I ignored them in the vehicle type classification, which resulted in an average 2% precision increase.

**a) J48**

|               | TP Rate | FP Rate | Precistion | Recall | F-Measure | MCC   | ROC Are | PRC Area | Class  |
| ------------- | ------- | ------- | ---------- | ------ | --------- | ----- | ------- | -------- | ------ |
|               | 0,950   | 0,714   | 0,884      | 0,950  | 0,916     | 0,301 | 0,570   | 0,855    | medium |
|         |     0,286 |   0,050 |   0,500 |     0,286 |   0,364    |  0,301 |   0,570  |   0,323 | heavy  |
| Weighted Avg. |0,851  |  0,615  |  0,827  |    0,851   | 0,833   |   0,301  |  0,570   |  0,776 |        |

Using *J48* I achieved an accuracy of 85.1%. I tried to optimize the result by tuning the *confidenceFactor* and the *minNumObj* parameter  in *Weka* but this only changed the outcome slightly, therefore I sticked to the default values: 0.25 for *confidenceFactor*, 2 for *minNumObj*.

**b) Naïve Bayes**

|               | TP Rate | FP Rate | Precistion | Recall | F-Measure | MCC   | ROC Are | PRC Area | Class  |
| ------------- | ------- | ------- | ---------- | ------ | --------- | ----- | ------- | -------- | ------ |
|               | 0,875  |  0,429  |  0,921  |    0,875  |  0,897  |    0,404  |  0,731  |   0,919    | medium |
|         |     0,571 |   0,125  |  0,444  |    0,571    |0,500  |    0,404  |  0,731  |   0,393 | heavy  |
| Weighted Avg. |0,830 |   0,383  |  0,850   |   0,830    0,838   |   0,404  |  0,731  |   0,841 |        |

Naive bayes has no parameters in *Weka*.

**c) kNN **

|               | TP Rate | FP Rate | Precistion | Recall | F-Measure | MCC   | ROC Are | PRC Area | Class  |
| ------------- | ------- | ------- | ---------- | ------ | --------- | ----- | ------- | -------- | ------ |
|               | 0,963  |  0,714  |  0,885   |   0,963   | 0,922  |    0,337  |  0,693  |   0,909    | medium |
|         |     0,286  |  0,038  |  0,571   |   0,286  |  0,381   |   0,337  |  0,693  |   0,352 | heavy  |
| Weighted Avg. |0,862 |   0,613  |  0,838  |    0,862 |   0,842  |    0,337  |  0,693  |   0,826 |        |

At first, I increased k to 10, which resulted in the misclassification of all heavy vehicles. The lack of heavy vehicles in the dataset made the kNN-algorithm struggle with false positives for large k in general. In the end I went for k = 3 to get the best possible result.

**d) Multilayer perceptron**

|               | TP Rate | FP Rate | Precistion | Recall | F-Measure | MCC   | ROC Are | PRC Area | Class  |
| ------------- | ------- | ------- | ---------- | ------ | --------- | ----- | ------- | -------- | ------ |
|               | 0,950  |  0,571  |  0,905   |   0,950  |  0,927   |   0,437  |  0,780  |   0,944    | medium |
|         |     0,429  |  0,050  |  0,600   |   0,429   | 0,500   |   0,437  |  0,780   |  0,443 | heavy  |
| Weighted Avg. |0,872  |  0,494  |  0,859   |   0,872  |  0,863   |   0,437  |  0,780  |   0,870 |        |

The multilayer perceptron performed best using the default parameters:

- learning rate: a small change did not change the performance, however a large increase/decrease worsened accuracy
- momentum: same as with learning rate
- training time: the accuracy was not getting better with increased training time but it got significantly worse when decreased

**Summary**

Measured on accuracy the multilayer perceptron performed best. However, the naïve Bayes hat the lowest false positive rate. 

 ##### 5. Classification of driving direction

