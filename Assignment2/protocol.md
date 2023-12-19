# Exercise PC Design & Development / Systems and Environments

## Assignment 2

**11908757 Stefan Haslhofer**

##### 1. Remarks

The data was acquired by walking the same street up and down to have similar recording conditions.

Python code: https://github.com/StefanHaslhofer/PervasiveComputing/tree/main/Assignment2

##### 2. Data recording

I recorded the acceleration over time on my mobile phone with an app called _Physics Toolbox Sensor Suite_.
The app allows access to the phone's built-in linear accelerometer and displays acceleration on x, y and z axis as well
as a total acceleration value.

<img src="PhysicsToolbox.jpg" alt="image" width="300" height="auto">

The data can be exported as _csv_. Each row contains a timestamp and all axis values plus the total acceleration.

##### 3. Pre-processing and Segmentation

At first, I plotted the recordings in the time domain and in the frequency domain to get a feeling for the data.
Looking at the time domain plot, one can see that it takes approximately 1 second to take a step. The frequency domain
confirms that frequencies in movement are (unsurprisingly) rather low. Hence, I choose to only consider frequencies
lower than 50Hz for further processing.
I also split the data recordings into windows of 5 seconds to get enough samples. I used a jumping window approach,
where windows do not overlap.

![leftHandTimePlot.png](leftHandTimePlot.png)

<img src="leftHandFreqPlot.png" alt="image" width="auto" height="400">

Above you can see the two plots:

1. x, y and z-axis plus total acceleration over time
2. frequency composition of total acceleration

##### 4. Feature extraction

Next, I implemented a python script to extract the features and save it to an _arff_-file

First, I calculated statistical features over the absolute signal values such as mean and variance for all axes
including the total acceleration. I also noticed that the phone accelerates faster in my pockets than in my hands.
Additionally, the frequencies recorded in my pockets seem to span over a wider spectrum compared to the recordings in my
hands which can be seen when comparing the following two plots with the left hand's plots in section 3:

![leftPocketTimePlot.png](leftPocketTimePlot.png)

<img src="leftPocketFreqPlot.png" alt="image" width="auto" height="400">

The significant difference in frequencies gave me reason to believe that I can extract viable information from the
frequency domain as well. I used scipy's fft implementation for that.

However, at first glance the frequencies only seem to make suitable distinction between hands and pockets but not
between left and right of each class.

The following table lists all extracted features with some remarks:

| feature               | remark                                                                                                                                                                    |
|-----------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| mean of x-axis        | mean acceleration of feet tends to be higher than those of hands, also right hand accelerates faster than left hand                                                       |
| mean of y-axis        | (see mean of x-axis)                                                                                                                                                      |
| mean of z-axis        | (see mean of x-axis)                                                                                                                                                      |
| mean of total Acc     | (see mean of x-axis)                                                                                                                                                      |
| var of x-axis         | variance of feet acceleration also tends to be larger than those of hands                                                                                                 |
| var of y-axis         | (see var of x-axis)                                                                                                                                                       |
| var of z-axis         | (see var of x-axis)                                                                                                                                                       |
| var of total Acc      | (see var of x-axis)                                                                                                                                                       |
| max x-axis Acc        | the maximum acceleration seems to be higher in the pockets (for all axes), also right hand accelerates faster than left hand                                              |
| max y-axis Acc        | (see var of x-axis)                                                                                                                                                       |
| max z-axis Acc        | (see var of x-axis)                                                                                                                                                       |
| max total Acc         | (see var of x-axis)                                                                                                                                                       |
| max x-axis freq       | max frequency is slightly higher for pockets                                                                                                                              |
| max y-axis freq       | (see max x-axis freq)                                                                                                                                                     |
| max z-axis freq       | (see max x-axis freq)                                                                                                                                                     |
| max total freq        | (see max x-axis freq)                                                                                                                                                     |
| max x freq energy     | dominant frequency of pockets has more energy than dominant frequency of hands, also right hand's dominant frequency has more energy than dominant frequency of left hand |
| max y freq energy     | (see max x freq energy)                                                                                                                                                   |
| max z freq energy     | (see max x freq energy)                                                                                                                                                   |
| max total freq energy | (see max x freq energy)                                                                                                                                                   |
| sum of x-axis energy  | overall energy of frequencies seem to differ between pockets, which is particularly helpful as I did not find a good distinction between left and right pocket yet        |
| sum of y-axis energy  | (see sum of x-axis energy)                                                                                                                                                |
| sum of z-axis energy  | (see sum of x-axis energy)                                                                                                                                                |
| sum of total energy   | (see sum of x-axis energy)                                                                                                                                                |
| q1 frequency          | frequency distribution differs widely between pockets and hands (see plots)                                                                                               |
| q2 frequency          | frequency distribution differs widely between pockets and hands (see plots)                                                                                               |

##### 5. Classification