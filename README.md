
# Housing Market Value Prediction by Machine Learning Algorithms

## 1. Abstract
Developing an accurate prediction model for housing property prices is always needed for socio-economic development and the well-being of citizens. In this paper, a diverse set of machine learning algorithms, such as linear regression, random forest regression, light regression, lasso regression, ridge regression, and others, are employed to predict housing prices using publicly available datasets. The Boston housing datasets for 2021 are obtained from the Boston Property Appraiser website. The records are publicly available and include the real estate and economic database, maps, and other associated information. The database is usually updated weekly according to the State of Massachusetts regulations. The Boston housing database records all residential and non- residential properties in the city. The Boston Globe reported in May 2021 that the Boston housing market is highly competitive, resulting in soaring costs. People are also looking for larger housing as the pandemic continues. Because most property managers and realtors cannot show their properties to multiple people, it becomes more difficult to find houses. This article aims to assist those individuals, realtors, and real estate agents by providing an approximate price for the home they are looking for. We choose to utilise a few fundamental machine learning ideas to assist in determining the optimal selling price for the home, given specific preferences on the number of rooms, location, style, and other details regarding the bath and kitchen. Then, the housing price prediction models using machine learning techniques are developed, and their regression model performances are compared. Finally, an improved housing price prediction model for assisting the housing market is proposed. Particularly, a house seller or buyer or a real estate broker can get insight into making better-informed decisions considering the housing price prediction. The overall purpose of this study is to build on earlier EDA (exploratory data analysis) work by developing predictive models that address our business problems. Finally, optimising the model's performance to achieve efficient results
## Keywords:
**Housing Market Forecasting, Machine Learning Algorithms, Linear Regression, Random Forest, Lasso, Light GBM Regression, Ridge Regression.**
## 2. Literature Review
### 2.1 Related Works
Ahmed et al. build a neural network-based housing market performance prediction model. This model is developed using a dataset of past market behavior to predict unforeseeable future market performance. The testing and validation of the model indicate that his Neural Net's prediction error is between -2 and +2 percent. 

Lim et al. build neural networks to anticipate the Singapore property market. They used the multilayer perceptron and the autoregressive integrated moving average for prediction. The model with the highest accuracy score is utilized for prediction, and the ANN model with the lowest mean square error (MSE) demonstrates that ANN is superior to other predictive techniques. Cokriging is a Multivariate Spatial Method for Predicting Housing Location Price that was developed by Chica et al. This approach estimates correlated spatial variables and creates interpolated maps of home values, giving appraisers and real estate brokers with information on house location pricing. During the experiment, home location price prediction values are computed utilizing isotopic and heterotopic cokriging techniques. Both approaches' results are compared, and the better method's forecast is picked.

Bahia et al. used an Artificial Neural network-based data mining model to the real estate market. FFB and CFBP network models were constructed throughout the investigation. Both of these models were trained using the Boston dataset, with regression value serving as the performance matrix. The CFBP prediction findings are the best, and the regression coefficient is 0.96; the research indicates that the CFBP prediction accuracy is 96%. Stevens et al.  predicted home prices using text mining. His price forecast includes pricing indications such as selling price, asking price, and price variation. This research demonstrates that the SGD classifier obtained the best results across all price metrics. Classification and regressions are performed using stemmed n-grams. R2 Matrix Prediction Performance Value is 0.303. Due to the complexity of the job, the research implies that both of these outcomes are satisfactory.

Nissan et al.  employed many algorithms to predict Montreal real estate market values. The research provides a model for predicting asking and selling prices based on characteristics such as location, area, number of rooms, proximity to police and fire stations, etc. They used several regression models to predict regression. Among these regression techniques are linear regression, SVR, kNN, regression Tree, and Random Forest Regression. The suggested prediction models have an error of 0.0985 for the Asking price and 0.023 for the Selling price.
Li et al.  used SVR, Support Vector Regression, for the first time to anticipate home prices in China in 2009. In 2016, Rafiei and Adeli also conducted research utilizing SVR  to assess if a property developer should halt development or commence a new project based on the forecast of future housing prices.

Using linear regression, Sifei Lu et al. (2017) established a sophisticated hybrid regression approach for predicting property prices. They utilized Gradient boosting, Ridge, and Lasso regression to create a hybrid model, and the outcome was substantial.

In 2018, Wang et al.  used about 30,000 house assessment price data from Virginia, United States, and concluded that Random Forest is more accurate than Linear Regression.

Mohd et al. (2019)  predicted housing prices in Petaling Jaya, Selangor, Malaysia using a variety of machine learning algorithms, including Ridge Regression, Random Forest, and Decision Tree, and determined that Random Forest is the most preferred algorithm in terms of overall accuracy, as measured by root mean squared error (RMSE).

Winky et al.  suggested Support Vector Machine (SVM), Random Forest (RF), and Gradient Boosting Machine (GBM) to analyze a data sample of around 40,000 housing transactions in 2020. Their research revealed that the SVM approach worked well since it can provide reasonably precise predictions within a little amount of time.

### 2.2 Research Gap and Contribution
In the available literature, just a limited level of interest has been devoted to the housing price prediction model, specifically to solving the issue using machine learning techniques. Several recognized documents were mentioned before. In addition, the majority of previous research viewed the housing market issue as classification difficulties in order to construct a classification model rather than a regression model. Consequently, the purpose of this work is to estimate the value of home prices utilizing machine learning methods and competitive regression models. Proposed is an enhanced ML-based approach that incorporates the anticipated target price binning variable as model features and considerably enhances the model's accuracy. Specifically, the accuracy of the model is improved by 10% compared to other modern machine learning approaches. Moreover, to the best of our knowledge, the Boston Property Appraiser's analyzed datasets have not been used in earlier studies of housing market price prediction difficulties.
## 3. Methodology
### 3.1 Data Pre-processing
#### 3.1.1 Data Collection
Author collected dataset from The Boston Property Appraiser website which provides all the 2021 Boston housing stats. The publicly accessible documents include the real estate and economic database, maps, and other related data. The Boston Housing Dataset is a derived from information collected by the U.S. Census Service concerning housing in the area of Boston MA.

#### 3.1.2 Data Cleaning
##### I.  Action on values in Mail_State Column. 
##### II. Action on values in BLDG_SEQ and NUM_BLDGS
##### III. Converting the currency values to float

#### 3.1.3 Data Preperation
The model includes both categorical (discrete) and numerical (continuous) characteristics. Preparing data for modelling guarantees that it is in the proper format. Eliminate any columns containing significant discrete values that impede the modelling process.Other This table has eight columns: **BLDG_VALUE, TOTAL_VALUE, 'CITY,' YR_BUILT, EXT_COND, KITCHEN_TYPE, HEAT_TYPE, and AC_TYPE**. To identify and incorporate important category variables, the label encoding will be used. If we had used a dummy variable technique, we would have been left with a large number of columns to deal with, since each feature had several values.
In addition, it would assist us in decoding the variable names at the conclusion. The subsequent step is to bucket (or bin) the continuous variables inside the dataset. We did so because binning enhances the accuracy of prediction models. It was necessary to bin **BLDG_VALUE** and **TOTAL_VALUE** features. For the **BLDG_VALUE** feature, we produced five bins containing all of the values dispersed within the BLDG_VALUE feature. Similarly, 10 bins were formed for **TOTAL_VALUE**.
We accomplished this because we need to deal with variables that forecast the property's value in a manner that is near to reality. We experimented with many bin sizes before deciding on this one. The first columns were eliminated since they were no longer required. It would have brought about multicollinearity. Next, we conducted a VIF test to assess the dataset's multicollinearity. We discovered several variables with VIF values over 10. We opted to exclude these variables since they would negatively impact the model's performance and cause prediction difficulties. The data will then be divided into train and test sets. This information will assist us in determining the model's Accuracy. We used a 70:30 ratio to input the model (train) with data, followed by testing. Thus, **83725 training records** and **35883 test record**s were acquired.

### 3.2 Modeling Frameworks
Utilizing the presented approach, the major objective of this case study is to forecast the home price for the supplied attributes in order to optimize the prediction accuracy. This housing issue may be categorized as both a regression and a categorization issue. Since the classification issue has been previously documented in the literature, this study examines several regression models with target variable binning that are used to housing market data to estimate the property price. The research framework for the home price prediction issue is shown in Figure 3.1. It consists of five primary blocks: data collection, data preparation, feature processing, model training, and model assessment. These diagram blocks are detailed in depth in the following sections.

![Housing Price Prediction Model](https://github.com/RakibulHSium/Housing-Price-Prediction/raw/7063c20387ab95d8003c5dd94ab6ee227c9ac1ff/Output/model.png)

Figure 3.1: Research framework for the housing price problem


## 4. Result and Analysis
### 4.1 Linear Regression model results on boston-2021 dataset:
After building a Linear Regression model, we identified critical categorical variables linear. From Figure 4.1, we see that the P-value of each variable; ** CITY, YR_BUILT, EXT_COND, KITCHEN_TYPE, and BLDG_VALUE_bins**, are essential variables. **KITCHEN_TYPE** has the highest co-efficient value, followed by **AC_TYPE** and **HEAT_TYPE**. According to this model, KITCHEN_TYPE is most important in predicting the estimated costs of the property in Boston.
Figure 4.2 and 4.3, graph shows the confusion matrix generated using Linear Regression. Using this confusion matrix, we calculated the model's Accuracy as 77.18% within 0.04299 seconds and other performance parameters shown in Figure 4.1, We opt for feature selection to optimize the model. We utilized Sequential Feature Selector to select backward.
Furthermore, we chose this strategy because it is commonly preferred. Here it shows us that a property price is affected by their **'KITCHEN_TYPE','BLDG_VALUE_bins'**. We used these results because they were better optimized.

![P-Value](https://github.com/RakibulHSium/Housing-Price-Prediction/raw/7063c20387ab95d8003c5dd94ab6ee227c9ac1ff/Output/Output1.png)

*Figure 4.1: P-value of each variable*

![C-matrix](https://github.com/RakibulHSium/Housing-Price-Prediction/raw/7063c20387ab95d8003c5dd94ab6ee227c9ac1ff/Output/output2.png)

*Figure 4.2: Linear Regression Confusion Matrix*

![C-matrix](https://github.com/RakibulHSium/Housing-Price-Prediction/raw/7063c20387ab95d8003c5dd94ab6ee227c9ac1ff/Output/output3.png)

*Figure 4.3: Linear Regression Confusion Matrix with labels*



### 4.2 Random Forest(Regressor) Results on Boston-2021 dataset:
Next, we performed Random Forest regressor modeling to uncover key categorical variables. Figure 4.4 displays the feature importance for each variable. BLDG_VALUE_bins, CITY, YEAR_BUILT, and KITCHEN_TYPE were all essential features this model has come up in predicting. According to this model, the BLDG_VALUE_bins is the most crucial factor in deciding the property's price in Boston. The confusion matrix created by this model prediction is shown in Figure 4.5. We computed the model's Accuracy of 69.23 percent within 1.8609 seconds execution time and other performance aspects using this confusion matrix displayed in Figures 4.5. 

<figure>
  <img src="https://github.com/RakibulHSium/Housing-Price-Prediction/raw/7063c20387ab95d8003c5dd94ab6ee227c9ac1ff/Output/output4.png" alt="Feature Importance of Each Variable">
  <figcaption>Figure 4.4: Feature Importance of Each Variable</figcaption>
</figure>

<figure>
  <img src="https://github.com/RakibulHSium/Housing-Price-Prediction/raw/7063c20387ab95d8003c5dd94ab6ee227c9ac1ff/Output/output6.png" alt="Random Forest Confusion Matrix with labels">
  <figcaption>Figure 4.5: Random Forest Confusion Matrix with labels</figcaption>
</figure>


### 4.3 LightGBM Regression Results on Boston-2021 dataset:
Last, we ran Gradient Boosting Machine (GBM) modeling. We observed that the modeling was quite time- consuming and opted to run the 'light' version. This LightGBM is significantly faster and delivers us the same results. We fitted the model where we uncovered key categorical variables. Figure 10 indicates the feature importance for each variable. CITY, KITCHEN TYPE, and BLDG VALUE bins were all crucial features this model has come up with in forecasting. According to this model, the CITY is the essential component in deciding the property's price in Boston. The confusion matrix obtained by this model prediction is displayed in Figure 8. We determined the model's Accuracy is 86.49 percent within 1.8609 seconds execution time and other performance factors using this confusion matrix given in Figures 4.6 and 4.7.

<figure>
  <img src="https://github.com/RakibulHSium/Housing-Price-Prediction/raw/7063c20387ab95d8003c5dd94ab6ee227c9ac1ff/Output/output7.png" alt="Confusion Matrix of LightGBM model">
  <figcaption>Figure 4.6: Confusion Matrix of LightGBM model</figcaption>
</figure>

<figure>
  <img src="https://github.com/RakibulHSium/Housing-Price-Prediction/raw/7063c20387ab95d8003c5dd94ab6ee227c9ac1ff/Output/output8.png" alt="LightGBM Regression Confusion Matrix with labels">
  <figcaption>Figure 4.7: LightGBM Regression Confusion Matrix with labels</figcaption>
</figure>


## 5. Model Regularization
How can we ensure that the obtained findings are stable and that the chosen characteristics will be constant across all generalized models? This protects the model from becoming overfit or underfit, as well as from producing inaccurate predictions. Ridge and Lasso Regression was used to verify that the findings were consistent or stable.
### Ridge Regression
We have attempted to apply the ridge regression model to the data, since the dependent variable is a continuous variable, as we go. The Ridge model is appropriate for elements with strong correlations. Our model is ideal for this dataset since the items we are analyzing are highly correlated. After executing the model, we determined that its Accuracy is 78.67%, which is greater. For the dependent variable, we may claim that the model worked well. We have undertaken backward feature selection to determine which features have the most influence via in-depth investigation. Through analysis, we determined that **KITCHEN_TYPE** and **BLDG_VALUE_bins** have a significant influence on the home price, since they are more significant factors when determining house pricing.
### Lasso Regression
Later, we attempted to implement the LASSO model, which assists in identifying the subset of predictors with the lowest error and highest Accuracy. It may help us determine which attributes have the most influence. In addition, after applying LASSO to the dataset, we discovered that the LASSO model performed very well with an Accuracy of 78.59%. The mean squared error for this model is 40.96%, placing it in second place behind the logistic regression model we developed. It is far less than the previous models we've conducted. After analyzing the model's performance, we can conclude that the model executed the backward feature selection really effectively. We discovered that both the LASSO and Ridge models produced the same significant feature, **KITCHEN_TYPE** and **BLDG_VALUE_bins**, but with significantly different Accuracy.















## Model Evaluation

<figure>
  <img src="https://github.com/RakibulHSium/Housing-Price-Prediction/raw/7063c20387ab95d8003c5dd94ab6ee227c9ac1ff/Output/output9.png" alt="Model Comparison Table">
  <figcaption>Figure 4.8: Model Comparison Table</figcaption>
</figure>

<figure>
  <img src="https://github.com/RakibulHSium/Housing-Price-Prediction/raw/7063c20387ab95d8003c5dd94ab6ee227c9ac1ff/Output/output10.png" alt="Model Comparison Bar Plot">
  <figcaption>Figure 4.9: Model Comparison Bar Plot</figcaption>
</figure>



## Conclusion
The Machine Learning methods used in this work were culled from published sources. Linear Regression, Random Forest, LightGBM Regression, Lasso, Ridge Regression are examples of machine learning models found in the literature. The objective of the study was to identify the optimal classification method for Housing Market Price Prediction using the Boston housing property datasets. All algorithms are subjected to an experiment, and the results are generated from the performance metrics chosen to address them. We have a solid understanding of the dataset and have prepared it through analysis. Based on the results of the predictive research, we have discovered that the housing prices are directly related to three factors: the value of the building, the location of the house, and the type of kitchen. When we look at the criteria, the building value and the location are the most important ones that can significantly impact the price. However, we cannot focus on that as a realistic prospect for the house prices. The type of kitchen offered in units is the most important factor that a property manager should consider. We saw that the kitchen type is driving the price since we categorized the kitchen values; we can say that having a well-furnished kitchen will significantly increase the property's price in Boston. If a property manager wants to make money and boost profit, he or she can change the kitchen type of houses priced lower than others to do so. The first thing that property management should look at is the type of kitchen available in the units. To generate income and raise the property's value, the manager can modify the type of kitchen available in the apartments that are less expensive than the others.
## References
Park, B., & Kwon Bae, J. (2015). Using machine learning algorithms for housing price prediction: The case of Fairfax County, Virginia housing data. Expert Systems with Applications, 42(6), 2928–2934. https://doi.org/10.1016/j.eswa.2014.11.040

Khalafallah, Ahmed et al. Neural network-based model for predicting housing market performance. Tsinghua Science and Technology. 2008, TUP,13,S1,325–328.

Lim, Wan Teng and Wang, Lipo and Wang, Yaoli and Chang, Qing. Housing price prediction using neural networks. 2016 12th International Conference on Natural Computation, Fuzzy Systems and Knowledge Discovery (ICNC-FSKD). 2016, IEEE,518–522.

Chica-Olmo, Jorge et al. Prediction of housing location price by a multivariate spatial method: Cokriging. Taylor & Francis. 2007, Taylor & Francis,29,1,91–114.

Bahia, Itedal Sabri Hashim and others. A Data Mining Model by Using ANN for Predicting Real Estate Market: Comparative Study. International Journal of Intelligence Science. 2013, Scientific Research Publishing,03,04,162.

Pow, Nissan and Janulewicz, Emil.Prediction of real estate property prices in Montreal. Repéré à urlhttp://rl.cs.mcgill.ca/comp598/fall2014/comp598_submission_99. pdf. 2014.

Li et al., “A SVR based forecasting approach for real estate price prediction” in International Conference on Machine Learning and Cybernetics, Hebei, 2009.

Wang, C. C., & Wu, H., “A new machine learning approach to house price estimation,” in New Trends in Mathematical Sciences, vol. 6, issue 4, pp. 165–171, 2018.

Sankranti Srinivasa Rao, “Stock Prediction Analysis by using Linear Regression Machine Learning Algorithm”, International Journal of Innovative Technology and Exploring Engineering (IJITEE), ISSN: 2278- 3075, Volume-9 Issue-4, February 2020.

Hochreiter S, Schmidhuber J. Long short-term memory. Neural Computation, 9(8): 1735-1780, 1997.

Cho K, Bahdanau D, Bougares F, Schwenk H and Bengio Y. Learning phrase representations using RNN encoder-decoder for statistical machine translation. In Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP 2014), 2014.

G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, and T. Liu “LightGBM: A Highly Efficient Gradient Boosting Decision Tree”. NIPS. 2017.

Yadav, A. and Vishwakarma, D.K., 2020. Sentiment analysis using deep learning architectures: a review. Artificial Intelligence Review, 53(6), pp.4335-4385.
## Authors

- [SIUM RAKIBUL HASAN](https://www.github.com/rakibulhsium)
- [Email](mailto:sium@nuist.edu.cn)


