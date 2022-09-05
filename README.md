# A Random Forest classification (supervised machine learning) that aims to predict the gender of smartphone users. 
* Written by: [Ivy Wu S.Y.](https://www.linkedin.com/in/ivy-wusumyi)
* Technologies: R, tidymodels, random forest, classification, ggplot2


## 1. Introduction
Classification is a supervised machine learning method aims to identify the category of a new observation given a labeled training data. In this section, a random forest classifier is built to predict the Gender of smartphone users. 


## 2. About the Data
The data contains demographics and personality data from a variety of Apple and Android users (n=529). These data contain the following variables:
| Variable       |                |
| -------------- | -------------- |
| Smartphone(OS) | iPhone/ Android|
| Gender         | Female/ Male   |
| Age            |                |
| Personality(HEXACO) | Honesty-Humility<br>Emotionality<br>Extraversion<br>Agreeableness<br>Conscientiousness<br>Openness|
|Avoidance Similarity||
|Phone as status object||
|Socioeconomic status||
|Time owned current phone (months)||


## 3. Methodology
<img src="/images/method.png?raw=true" width="550"><br/>
The dataset is split into a training set (80%) and a test set (20). All variables are included in the first modeling to facilitate the backward feature selection in the later stage. The modeling process starts with fitting a random forest model and cross-validate 10 folds with 25 random grid searches. Instead of calculating overall accuracy, Kappa statistic is used to compare the performance of each model as it considers the imbalance of class distribution. The selected model will be used for predicting the out-of-sample data and evaluated with different performance matrices. A finetune process composes of feature selection, hyperparameter tuning or model selection with other performance matrices is an optimal step to be carried out at the end for model improvement.

## 4 Fitting Models and Model Selection
During the modeling process, a variety of models are obtained with different tunings of parameters. There are three parameters namely the mtry (the number of randomly sampled predictors at each split), trees (the number of trees contained in the ensemble) and min_n (the minimum number of data points in a node that are required to be split further) need to be decided for the Random Forest model. An auto-tunning function in R enables the algorithm to auto-tune those parameters and return the model performance of your choice (Kappa statistic is used in this case). <br/>
<img src="/images/models.png?raw=true" width="550"> <br/>
To choose between them, Kappa statistic is used to determine how well the models perform. As illustrated above, Model 12 is the best in terms of Kappa statistic (0.435) with parameters mtry = 11, trees = 1783 and min_n = 13. As such, model 12 is selected for the out-of-sample prediction. 

## 5. Model Evaluation
After finalizing the model choice, it is used to predict the out-of-sample data and carry out model evaluation. Below figure shows the confusion matrix for the model prediction on 106 observations. The overall accuracy of the model is quite high with 72.6% of predictions being correct. Essentially, the model is classifying most of the “female” correctly but fewer “male” can be correctly classified. This is also visible by a lower sensitivity value of class “male” (52.94%) compared to that of “female” (81.94%), leading to a balanced accuracy of 67.4%. As expected, the Kappa statistic is lower than that of the training set, with a value of 0.36 indicating a fair agreement.<br/>
<img src="/images/confusion_matrix.png?raw=true" width="550"><br/>

ROC and AUC is another useful evaluation matrix as it does not depend on the class distribution. By plotting the true-positive rate against the false-positive rate with a baseline obtained with a random classifier, the plot reveals an AUC value of 0.75 which suggests the model has a moderate capability of distinguishing between classes. <br/>
<img src="/images/ROC.png?raw=true" width="350"><br/>


## 6. Model Improvement: Backward Feature Selection
There are multiple ways to finetune the model for model improvements. A typical method is to perform feature selection by selecting the most important variables and eliminating those redundant features that might increase the risk of overfit to noise in the data. A variable importance score is obtained from the tree decision, showing that “Smartphone” does not contribute much in determining the target outcome. As such, a second random forest model is built with the removal of “Smartphone” variable. <br/> 
<img src="/images/variable_importance.png?raw=true" width="550"><br/> 
With the same modeling procedure, the performance of the second model, however, does not improve as expected. The overall accuracy and kappa statistic was reduced to 69.8% and 0.25 respectively, indicating the new model with subset features was inferior to the prior model. The reason was probably due to the ensemble nature of random forest with random feature extraction during the splitting of nodes. The randomization leads to a similar feature selection process and hence causing less effect on model improvement. Other tunning options such as grid search optimization should be considered in this case. <br/> 
<img src="/images/secondRF_result.png?raw=true" width="350"><br/> <br/> 

## 6. Limitations of Random Forest Classification:
1.	The algorithm can be computationally intensive as it generates hundreds of trees.
2.	A forest is less interpretable than a single decision tree. The algorithm is a black-box model since it is impossible to gain a full understanding of the decision process by examining all individual trees. 

### To learn more about Ivy Wu S.Y., visit her [LinkedIn profile](https://www.linkedin.com/in/ivy-wusumyi)

All rights reserved 2022. All codes are developed and owned by Ivy Wu S.Y..
