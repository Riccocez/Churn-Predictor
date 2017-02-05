
# coding: utf-8

# # Churn Predictions of taxi rides randomly generated
# 
# 
# This script shows the performance of 3 churn predictors of hypothetical records from 1000 taxi drivers generated randomly. The goal is to build a Churn predictor capable of cluster these drivers into the 3 following classes:
# 
#     1. Low Churn
#     2. Medium Churn
#     3. High Churn
#  
# A driver is considered to be churned when he has not accepted any passenger in 30 days. The feature 'Lapsed days' indicates the number of days a driver has not accepted any passenger. We include around 30 features comprising descriptive information related to the factors we considered relevant for predicting churn drivers. These features can be sorted in three main topics: 1) Data related to the driver or passengers 2) Data related to the taxi app and 3) Data related to external aspects. The approach we used to perform the predictions is multiClass Logistic Regression.  
# 
# The first model was built setting up the following thresholds for each class:
#     -Low Churn: Drivers from 0 up to 7 days lapsed
#     -Medium Churn: Drivers from 7 up to 20 days lapsed
#     -High Churn: Drivers from 20 up to 51 days lapsed. 50 days is the maximum lapsed day registered
# 
# The second model was built considering the same thresholds that first model. However, drivers were labeled randomly in order to prove the relevance of the constrains proposed for days lapsed.
# 
# The third model considered the following thresholds for each class:
#     -Low Churn: Drivers from 0 up to 10 days lapsed
#     -Medium Churn: Drivers from 10 up to 17 days lapsed
#     -High Churn: Drivers from 17 up to 51 days lapsed. 50 days is the maximum lapsed day registered
# 
# Relevant findings indicate that the first model proves to have a better performance when predicting the 3 Churn classes. Despite its performance is lower when predicting whether a driver belongs to the Lower or Medium Class, its performance is impresive when detecting High Churn drivers. Further details regarding the performance of this model can be found below.
# 
# The second model is the worst among the three models, and it is reasonable since we labeled randomly all of the drivers in our dataset. This definitely complicates to our model understanding the behaviour of our drivers according to the features we provided. It's very likely that features and labels used during the training stage had strong inconsistencies, which caused the model didn't understand clearly the correlation among the data provided.
# 
# Finally, the third model doesn't provide a much better performance than the second model. Drivers were classified most of the time as a Low or High Churn drivers. If we were to implement this model in production, we could have been facing critical issues related to churn. Based on this model, we could be assuming most of the costumers will be using our app for a long time or that they simply will leave our app very soon. An this might actually not be reflecting the reality. So we consider this is a powerful reason for not considering the third model as the preferred model to predict Churn clusters. 

# In[1]:

get_ipython().magic(u'matplotlib inline')
get_ipython().magic(u"config InlineBackend.figure_format='retina'")


# In[2]:

from Predictor import Predictor
from Dataset import Dataset as ds
import Randomness as rnd
from Visualization import Visualization


# In[3]:

def build_predictor(minThresholds = [],maxThresholds = [],generate = False, 
                    train = True, fileName = 'predictor_best',random_labels = False,
                   matrix_labels = ['Low','Low','Medium','high']):
    """
    * Generates a predictor with a Dataset, a Predictor and a Visualization object 
    *
    * minThresholds: A list containing the minimum threshold values for each of the classes 
    * considered during labelling the dataset 
    * maxThresholds: A list containing the maximum threshold values for each of the classes 
    * considered during labelling the dataset
    * train: Indicates whether is desired to train a new model
    * fileName: The name of the file where a pretrained model is saved
    * random_labels:  Defines whether to generate a set of random labels for the y_labels
    * of training and testing data
    * matrix_labels: The set of labels to be shown in the plot
    """
    
    #Build random dataset
    dataset = ds()
    dataset.create_dataset(generate = generate, random_labels = random_labels, 
                       minThresholds = minThresholds, maxThresholds = maxThresholds
                      )
    
    #Build Churn Predictor
    predictor = Predictor(x_test = dataset.x_test,y_test = dataset.y_test,
    x_train = dataset.x_train,y_train = dataset.y_train)

    predictor.load_model(train = train,fileName=fileName)
    predictor.test_model()

    # Generate Visualizations of Churn Predictions
    visualization = Visualization()

    visualization.build_confussion_matrix(dataset.y_test, predictor.get_predictions())
    visualization.plot_matrix(matrix_labels = matrix_labels)
       
    return dataset,predictor,visualization


# ## Model 1: Labeled Data according to first proposed thresholds

# Another interesting behaviour of this model is that there is a tendency of having more false positives than false negatives. This model prefers to cluster a real "Medium Churn driver" as a "High Churn driver", as well as a real "Low Churn driver" as a "Medium Churn driver". We consider this might be helpful for the churn predictor of our app, since in a real situation, we might be interested in detecting highly potential drivers to churn. Being able to detect drivers' with very high potential to churn before they actually do it, give us a chance to look for diferent strategies to reduce their intention to churn or to reduce their impact on further customers. For instance, we could have enough time to launch an interesting offer to them, or (for reducing affecting other customers) by avoiding recommending our most valious passengers to this group of drivers. 

# In[16]:

dataset_01,predictor_01,visualization_01 =  build_predictor(
                                                minThresholds = [0,7,20], maxThresholds = [7,20,51],
                                                generate = False,train=False, fileName = 'best_model', 
                                                matrix_labels = ['Low','Low','Medium','high'])
#Save model in a local file if desired
#predictor_01.save_model('best_model')


# # Model 2: Labeled Data according to random values

# As explained above, this model has one of the worst performance of this analysis. It's clear that the model might have found
# strong inconsistencies in the data, which very likely caused a poor performance when predicting the churn classes desired.
# 

# In[17]:

dataset_02,predictor_02,visualization_02 =  build_predictor(
                                                minThresholds = [0,7,20], maxThresholds = [7,20,51],
                                                generate = False,train=False, fileName = 'average_model', 
                                                random_labels = True, matrix_labels = ['Low','Low','Medium','high'])
#Save model in a local file if desired
#predictor_02.save_model('average_model')


# # Model 3: Labeled Data according to a more streched threshold

# The third model, shows strong assumptions in the clustering process. It's very difficult for the model to detect Medium Churn drivers, which might put on risk the expansion of our app if we were implementing this model in our platform. We just simply wouldn't have enough time to react when drivers would be churning. The accuracy of High churn drivers is very interesting, however, it might be very difficult for us to understand what are the causes that make the drivers churn.

# In[18]:

dataset_03,predictor_03,visualization_03 =  build_predictor(
                                                minThresholds = [0,10,17], maxThresholds = [10,17,51],
                                                generate = False,train=True, fileName = 'predictor_best', 
                                                matrix_labels = ['Low','Low','Medium','high'])
#Save model in a local file if desired
#predictor_03.save_model('second_best_model')


# In[ ]:



