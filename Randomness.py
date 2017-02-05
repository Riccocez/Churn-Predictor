
# coding: utf-8

# In[6]:

import faker
import random


# In[26]:

class Randomness(object):
    
    def __init__(self, numDays = 1, yrsBefore = 1 , minValue = 0, maxValue = 0):
        """
        * Generates Random values/dates according to a minimum and maximum threshold 
        """
            
        self.numDays = numDays
        self.yrsBefore = yrsBefore
        self.minValue = minValue
        self.maxValue = maxValue
        self.fake = faker.Faker()
        
    def random_value(self, minValue, maxValue,floatValue=False):
        """
        * Generates a float/integer random value
        """
        
        if not floatValue:
        
            return random.randint(minValue, maxValue)
    
        elif floatValue:
        
            return round(random.uniform(minValue, maxValue),3)
          
    def random_date(self):
        """
        * Creates random date from yrsBefore until now 
        """
        return self.fake.date_time_between(start_date='-'+ str(self.yrsBefore) +'y',end_date='now').isoformat()
    
        
    def random_dates(self, yrsBefore):
        """
        * Creates n random timestamp values 
        """
        self.yrsBefore = yrsBefore

        return self.random_date()
    
    

