# Improving Covid-19 Prediction Through Variational Bayes For Learned Class Weighting
Current Covid-19 datasets are rife with class imbalance. This study identifies the consequences of such imbalance on
Covid-19 diagnosis tasks, as well as presenting both existing and novel solutions to these problems. Firstly, class-based upsampling and loss function weighting are used. Combined with data augmentation, these strategies achieve a much greater
performance on the Covid-19 minority class. However, this comes at the cost of lower accuracy in majority classes. As such,
a novel adaptation of the Variational Bayes for Active Learning thesis is used to predict dynamic, per-sample classification
difficulty scores for upsampling and loss function weighting. This method improves minority class prediction while improving overall performance, alleviating issues of the existing methods. In addition, the efficacy of supervised and unsupervised
pretraining as a class imbalance strategy is explored. The findings show that pretraining may act as an orthogonal strategy that
improves classification performance when combined with other strategies.


##Dataset Distribution

![](https://github.com/Jack742/Improving-Covid-19-Prediction-Through-Variational-Bayes-For-Learned-Class-Weighting/blob/main/Results%20and%20Paper/Dataset%20Distribution.png)
