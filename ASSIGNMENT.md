# NPFL128-Project

# Exploring linguistics and deep learning techniques to enhance football predictions

*Description:* I have an ongoing project in the football fantasy realm. In this project I collected data, preprocessed it and then ran various experiments using many traditional ML models + a little bit of MLP, where I didn’t put much effort into the architecture, hyperparameters and even avoiding overfitting. Two prediction tasks in the project are player performance prediction (regression) and an underpriced player identification (clustering and selecting). I would like to see how the techniques we talked/will talk about in this course would enhance these predictions. I think I will focus solely on the regression part.

The inputs to my models in both of these tasks were statistics, therefore numerical. Making progress past a certain threshhold in the root mean square error - RMSE (or other metrics) was hard and I think that the problem lies in the data, which is underrepresentative of the complicated reality of sports. The hypothesis is that incorporating textual data (sentiment, articles, opinions) and using more advanced deep learning models could improve the results.

*Selected papers out of which I will create a summary:*

· Incorporating textual data to sports projects

Deep Learning Contextual Models for Prediction of Sport Events Outcome from Sportsmen Interviews https://acl-bg.org/proceedings/2019/RANLP%202019/pdf/RANLP142.pdf

Innovative Approaches in Sports Science—Lexicon-Based Sentiment Analysis as a Tool to Analyze Sports-Related Twitter Communication https://www.mdpi.com/2076-3417/10/2/431

Comprehensive Analysis of Classifier to Identify Sentiment in Football Specific Tweets https://eudl.eu/doi/10.4108/eai.16-5-2020.2304099

· Deep learning for sports

A Deep Learning Framework for Football Match Prediction https://ieeexplore.ieee.org/abstract/document/9740760?casa_token=i-brzCI6XwoAAAAA:WaF8OlaDCD303OJl4apErX4Pwx7zZBvSbwlAhDy4Hbjs9NLImSjDWs21ruEZFeQzojt_qaltIeg

Sports match prediction model for training and exercise using attention-based LSTM network https://www.sciencedirect.com/science/article/pii/S2352864821000602

*Related work:* My bachelor’s thesis https://dspace.cuni.cz/handle/20.500.11956/183090?locale-attribute=en

*Data:* datasets used in my bachelor’s thesis, those datasets that occur in the papers (if applicable), data from twitter API such as described in this notebook Tweets of Top European Football (Soccer) Clubs | Kaggle, possibly self-scrape data from other sites such as reddit 

*Algorithm:*

1) processing the textual data and creatings features that capture the football reality well

2) extending my existing datasets with this data

3) proposing deep learning architectures based on the papers read

4) running experiments and evaluating against a baseline

*Evaluation plan:* The baseline to evaluate against will be my results from the previous project. I expect the text enhanced dataset, along with improving machine learning models by using deep learning architectures, to outperform the existing implementation.
