import os,sys
from src.logger import logging
from src.exception import CustomException
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

class DataTransformation():
    def initiate_data_transformation(self,trainset,testset):
        try:
            logging.info('Data Transformation Started.')
            train_df = pd.read_csv(trainset)
            test_df = pd.read_csv(testset)
            logging.info('Reading of train and test data completed.')
            target_column_name = 'LeaveOrNot'
            cat_column_name = ['Education','City','Gender','EverBenched']
            numerical_column_name = ['JoiningYear','PaymentTier','Age','ExperienceInCurrentDomain','LeaveOrNot']

            logging.info("Applying the encoding object on training and testing dataframe")
            train_df['Education'] = train_df['Education'].replace({'Bachelors':1,'Masters':2})
            test_df['Education'] = test_df['Education'].replace({'Bachelors':1,'Masters':2})

            train_df['City'] = train_df['City'].replace({'New Delhi':1,'Bangalore':2,'Pune':3})
            test_df['City'] = test_df['City'].replace({'New Delhi':1,'Bangalore':2,'Pune':3})

            train_df['Gender'] = train_df['Gender'].replace({'Male':1,'Female':2})
            test_df['Gender'] = test_df['Gender'].replace({'Male':1,'Female':2})

            train_df['EverBenched'] = train_df['EverBenched'].replace({'No':0,'Yes':1})
            test_df['EverBenched'] = test_df['EverBenched'].replace({'Yes':1,'No':0})

            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]


            train_arr = np.c_[
                np.array(input_feature_train_df),np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                np.array(input_feature_test_df),np.array(target_feature_test_df)
            ]
            logging.info('Transformation Complete.')
            return(train_arr,test_arr)
        
        except Exception as e:
            raise CustomException(e,sys)