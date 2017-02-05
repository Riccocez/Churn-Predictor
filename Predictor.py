
# coding: utf-8

# In[1]:

from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib


# In[2]:

class Predictor(object):
    
    def __init__(self,x_train = None, y_train = None, x_test = None, y_test = None):
        """
        * Generates a Predictor object with training and testing data 
        * Generates self.predictor - the predictor model
        """
        
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.predictor = None
        self.predictions = None
    
    def load_model(self, train = False,fileName = 'best_predictor'):
        """
        * Loads a predictor model into the Predictor object
        * When train:False a pretrained model is loaded. Otherwise, a new model is
        * trained with the training data of the Predictor object
        *
        * train: Indicates whether is desired to train a new model
        * fileName: The filename where the model is saved
        """
        
        if not train:
            self.predictor = joblib.load(fileName + '.pkl')
        else:
            self.train_model()
        return
    
    def set_train_test_data(self,x_train,y_train,x_test,y_test):
        """
        * Sets the training and testing data of the Predictor object
        *
        * x_train: the records of the training data
        * y_train: the true label of each record in x_train
        * x_test: the records of the testing data
        * y_test: the true label of each record in x_test
        """
        
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        
        return
        
        
    def train_model(self):
        """
        * Trains the Predictor object using Logistic Regression
        """
        
        self.predictor = LogisticRegression().fit(self.x_train, self.y_train)
        return
    
    def test_model(self):
        """
        * Predicts the set of responsed resulting of testing the model with 
        * the testing data contained in the Predictor object
        """
        
        self.predictions = self.predictor.predict(self.x_test)
        return
    
    def get_predictions(self):
        """
        * Returns the set of predictions performed by self.predictor
        """
        
        return self.predictions
    
    def get_y_test(self):
        """
        * Returns the true labels of the testing data contained in the Predictor object
        """
        
        return self.y_test
        
    def save_model(self,fileName):
        """
        * Saves the model into a local file to perform further predictions in new/contained testing data.
        *
        * fileName: The filename where the model is saved
        """
        
        joblib.dump(self.predictor, fileName + '.pkl') 
        
        return
        

