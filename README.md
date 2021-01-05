# Improving Covid-19 Prediction Through Variational Bayes For Learned Class Weighting
Current Covid-19 datasets are rife with class imbalance. This study identifies the consequences of such imbalance on
Covid-19 diagnosis tasks, as well as presenting both existing and novel solutions to these problems. Firstly, class-based upsampling and loss function weighting are used. Combined with data augmentation, these strategies achieve a much greater performance on the Covid-19 minority class. However, this comes at the cost of lower accuracy in majority classes. As such, a novel adaptation of the Variational Bayes for Active Learning thesis is used to predict dynamic, per-sample classification
difficulty scores for upsampling and loss function weighting. This method improves minority class prediction while improving overall performance, alleviating issues of the existing methods. In addition, the efficacy of supervised and unsupervised pretraining as a class imbalance strategy is explored. The findings show that pretraining may act as an orthogonal strategy that improves classification performance when combined with other strategies.


## Dataset Distribution
The [CovidX dataset](https://github.com/lindawangg/COVID-Net/blob/master/docs/COVIDx.md) was used for training all networks. At the time of producing this project, the dataset was heavily imbalanced:

![](https://github.com/Jack742/Improving-Covid-19-Prediction-Through-Variational-Bayes-For-Learned-Class-Weighting/blob/main/Results%20and%20Paper/Dataset%20Distribution.png)

As a result, network performance on the minor class of Covid-19 samples was poor.

## Existing Methods
The first strategies leveraged were data augmentation (adding perturbations to samples to increase variance within the dataset), upsampling (sampling the minority-class at a greater proportion to its size) and loss-function weighting (applying greater weights to the loss function for the minority class. By combining all three strategies, the performance of network on the minority class improved. However, the performance on the majority classes was worstened (in other words, the network simply predicted Covid-19 more frequently, instead of actually learning the features of Covid-19).

## Variational Bayes
Through using a variational autoencoder and Bayesian statistics, it is possible to estimate the difficulty of a classification for each sample [1](https://arxiv.org/abs/2003.11249). This project took this notion and presents a novel application of the methodolgy for dynamic, real-time and per-sample loss-function weighting and upsampling. This created a more robust network, that improved on minority-class samples without sacrificing majority-class performance.

![](https://github.com/Jack742/Improving-Covid-19-Prediction-Through-Variational-Bayes-For-Learned-Class-Weighting/blob/main/Results%20and%20Paper/VaBAL_against_other_methods_bar.png)

## MoCo and Pretraining
Finally, the efficacy of pretraining the CNN on large existing databases was explored (specifically, traditional pretraining on ImageNet and [Momentum Contrast Learning](https://arxiv.org/abs/1911.05722) against the Instagram 1bn dataset). This resulted in a signficantly stronger performance on minority-class samples, but lead to an overall reduction in performance. The intuition is that the generic feature extractor is not biased by the dataset imbalance, however the lack of training against chest X-ray images results in worse performance overall.

![](https://github.com/Jack742/Improving-Covid-19-Prediction-Through-Variational-Bayes-For-Learned-Class-Weighting/blob/main/Results%20and%20Paper/Pretrained_barchart_comparison.png)
