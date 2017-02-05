
# coding: utf-8

# In[14]:

import pandas as pd
import numpy as np
from Randomness import Randomness
import random


# In[13]:

class Dataset(object):
    
    def __init__(self, instances = 1000, generate = False, data = None, columns = None, dataset = None):
        """
        * Generates a Dataset object of n instances with random data
        * Generates self.dataset - the dataset stored as dataframe
        *
        * instances: The number of instances desired to generate for the dataset
        * generate: Define whether to generate a new dataset or not
        * data: The raw data of the Dataset object
        * columns: The list of column names required for creating the Dataframe of dataset
        * dataset: The dataset of the Dataset object stored as a Dataframe
        """
        self.instances = instances
        self.data = data
        self.generate = generate
        self.columns = columns
        self.dataset = dataset
        self.y_train = None
        self.x_train = None
        self.x_test = None
        self.y_test = None
        self.y_name = None
    

    def generate_dataset(self,instances = 1000,generate = False):
        """
        * Generates a new Dataset of n instances when generate:True
        * Reads a new Dataset already created when generate:False
        *
        * instances: The number of instances desired to generate for the dataset
        * generate: Define whether to generate a new dataset or not
        """ 
        if not generate:
            
            self.get_dataframe()
            
        else:    
            
            self.data = []
            
            for i in range(0,instances):
                
                randomness = Randomness()
            
                self.data.append(
                    [  
                        randomness.random_dates(yrsBefore=1), randomness.random_value(1,3), randomness.random_value(1,10), randomness.random_value(1,15,True),             
                        randomness.random_value(200,1000,True), randomness.random_value(1,26), randomness.random_value(50,1000), randomness.random_value(10,1000,True),              
                        randomness.random_value(1100,10000,True), randomness.random_value(0,25), randomness.random_value(0,1980,True), randomness.random_value(200,10000,True),             
                        randomness.random_value(10,100,True), randomness.random_value(110,20000), randomness.random_value(1,200), randomness.random_value(1,100),               
                        randomness.random_value(1,50), randomness.random_value(1,5), randomness.random_value(20,100,True), randomness.random_value(1,400,True),            
                        randomness.random_value(2,35,True), randomness.random_value(10,50,True), randomness.random_value(30,70), randomness.random_value(500,15000),            
                        randomness.random_value(1000,100000), randomness.random_value(1000,4000), randomness.random_value(1,5,True), randomness.random_value(10,80,True)
                    ]
                                )
            
            self.to_dataframe(writeFile = generate)
            
        
        return
    
    def to_dataframe(self, writeFile = False):
        """
        * Maps data to a Dataframe
        * The dataset can be saved in a local file if writeFile:False
        *
        * writeFile: Defines whether to save the dataset information into a local json file called "dump_dataset.json"
        """     
        self.define_colNames()
        self.dataset = pd.DataFrame(self.data, columns = self.columns)
        
        if writeFile:
            
            self.dataset.to_json('dump_dataset.json')
            
        return
    
    def get_dataframe(self):
        """
        * Gets a new dataset already created as a dataframe
        """
        self.dataset = pd.read_json('dump_dataset.json')
        
        return
    

    def define_colNames(self):
        """
        *  Defines the name of each column in the dataframe
        """
        self.columns = ['Ts_start_session', 'os_platform', 'car_brand','session_time',
               'cum_session_time', 'passengers', 'cum_passengers', 'revenue',
               'cum_revenue', 'rides', 'trip_distance', 'cum_trip_distance',
               'driver_average_rides', 'cum_rides', 'active_days', 'retained_days',
               'lapsed_days', 'country', 'ARPDAD', 'ARPP',
               'ATD', 'average_session_time',
               'average_rides_driver', 'total_requested_rides',
               'DAP','DAD','Rank','churn_rate']
    
        return 
      
    def get_tsfeats(self):
        """
        * Extracts and includes features derived from timestamp instances into the self.dataset
        """
        dt_index = pd.DatetimeIndex(self.dataset['Ts_start_session'])
        self.dataset['dayofweek'] = dt_index.dayofweek
        self.dataset['dayofyear'] = dt_index.dayofyear
        self.dataset['dayofweek'] = dt_index.dayofweek
        self.dataset['month'] = dt_index.month
        self.dataset['hour'] = dt_index.hour
        self.dataset['minute'] = dt_index.minute
    
        return     

    def clean_data(self):
        """
        * Removes duplicates from dataframe
        """
        self.dataset = self.dataset.drop_duplicates()
        return  

    def remove_field(self,fieldName):
        """
        * Removes fieldName column from the dataframe
        *
        * fieldName: The name of the column desired to remove from the dataframe
        """
        if fieldName in self.dataset:
            del self.dataset[fieldName]
            self.columns = self.dataset.columns
        return 
    
    
    
    def set_threshold(self,minThresholds,maxThresholds):
        """
        * Sets a list of min and max Thresholds required to label the dataset in the Dataset object
        *
        * minThresholds: A list containing the minimum threshold values for each of the classes 
        * considered during labelling the dataset 
        * maxThresholds: A list containing the maximum threshold values for each of the classes 
        * considered during labelling the dataset
        """
        
        self.minThresholds = minThresholds
        self.maxThresholds = maxThresholds
        
        return
    
    def generate_labels(self,minThresholds,maxThresholds):
        """
        * Generate the set of labels in the dataset according to the min and max Thresholds 
        *
        * minThresholds: A list containing the minimum threshold values for each of the classes 
        * considered during labelling the dataset 
        * maxThresholds: A list containing the maximum threshold values for each of the classes 
        * considered during labelling the dataset
        """
        
        self.set_threshold(minThresholds,maxThresholds)
        self.dataset['Churn'] = [self.get_label(instance) for instance in self.dataset['lapsed_days']]
        
        return
    
    def get_label(self,instance):
        """
        * Returns the class to which an instance belongs to based on the minimum and maximum thresholds
        *
        * instance: the instance to be labelled 
        """
        
        
        # Low Churn Class
        if instance >= self.minThresholds[0] and instance < self.maxThresholds[0]:
            return int(0)
        # Medium Churn Class
        elif instance >= self.minThresholds[1] and instance < self.maxThresholds[1]:
            return int(1)
        # High Churn Class
        elif instance >= self.minThresholds[2] and instance < self.maxThresholds[2]:
            return int(2)
        else:
            return int(2)
    
    def build_train_testdata(self,testsize = 0.3,fieldName='Churn'):
        """
        * Builds the train and test data considering the testsize desired and the fieldName
        * that contains the output data
        *
        * testsize: The size of the test data we want to consider from the dataset contained 
        * fieldName: The field that contains the output data in the dataset
        """
        
        size = int(len(self.dataset)*(testsize))
        
        self.get_ylabels(fieldName,size)
        
        self.x_train = self.dataset[self.columns.values][0:-size].values
        self.x_test = self.dataset[self.columns.values][-size:].values
        
        
        return
        
    
    def get_ylabels(self,fieldName,size):
        """
        * Gets the output labels from the dataset and stores them into the Dataset object.
        * The field containing the ouput labels is removed from the features to be used when
        * training a given model.
        *
        * fieldName: The field that contains the output data in the dataset
        * size: The size of the test data we want to consider from the dataset contained
        """
        
        self.y_name = fieldName
        self.y_train = self.dataset[fieldName][0:-size].values
        self.y_test = self.dataset[fieldName][-size:].values
        
        self.remove_field(fieldName)

        return
    
    def create_dataset(self,generate=False,random_labels = False, minThresholds=[],maxThresholds=[],testsize=0.3,fieldName='Churn'):
        """
        * Main method from Dataset object. Performs the steps required to create a dataset
        * considering a set of given conditions:
        *
        * generate: Defines whether to generate a new dataset or not
        * random_labels:  Defines whether to generate a set of random labels for the y_labels
        * of training and testing data
        * minThresholds: A list containing the minimum threshold values for each of the classes 
        * considered during labelling the dataset 
        * maxThresholds: A list containing the maximum threshold values for each of the classes 
        * considered during labelling the dataset
        * testsize: The size of the test data we want to consider from the dataset contained 
        * fieldName: The field that contains the output data in the dataset 
        """
        
        self.generate_dataset(generate=generate)
        self.define_colNames()
        self.get_tsfeats()
        self.remove_field('Ts_start_session')
        
        if random_labels:
            self.create_random_labels()
        else:
            self.generate_labels(minThresholds=minThresholds,maxThresholds=maxThresholds)
        self.build_train_testdata(testsize=testsize,fieldName=fieldName)
        return
    
    def create_random_labels(self):
        """
        * Generates random labels for each instance of the "Lapsed days" column of the dataset
        *
        """
 
        randomness = Randomness()
        
        self.dataset['Churn'] = [randomness.random_value(0,2) for instance in self.dataset['lapsed_days']]
        
        return    
        


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



