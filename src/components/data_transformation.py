import sys
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object
#from src.components.data_transformation import DataTransformationConfig
#from src.components.data_transformation import DataTransformation


@dataclass
class DataTransformationConfig:
    preprocessor_obj_path= os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

# this function is responsible for data transformation
    def get_data_transformer_object(self):
        try:
            logging.info("Data Transformation initiated")

            # ============================
            # 🔢 NUMERICAL COLUMNS
            # ============================
            numerical_columns = ['temperature']

            # ============================
            # 🟢 ORDINAL CATEGORICAL COLUMNS
            # ============================
            ordinal_columns = [
                'age', 'education', 'income', 'Bar', 'CoffeeHouse',
                'CarryAway', 'RestaurantLessThan20', 'Restaurant20To50'
            ]

            ordinal_mappings = [
                ['below21', '21', '26', '31', '36', '41', '46', '50plus'], 
                ['Some High School', 'High School Graduate', 'Some college - no degree',
                 'Associates degree', 'Bachelors degree', 'Graduate degree (Masters or Doctorate)'],
                ['Less than $12500', '$12500 - $24999', '$25000 - $37499', '$37500 - $49999',
                 '$50000 - $62499', '$62500 - $74999', '$75000 - $87499', '$87500 - $99999', '$100000 or More'],
                ['never', 'less1', '1~3', '4~8', 'gt8'],
                ['never', 'less1', '1~3', '4~8', 'gt8'],
                ['never', 'less1', '1~3', '4~8', 'gt8'],
                ['never', 'less1', '1~3', '4~8', 'gt8'],
                ['never', 'less1', '1~3', '4~8', 'gt8']
            ]

            # ============================
            # 🔵 NOMINAL CATEGORICAL COLUMNS
            # ============================
            nominal_columns = [
                'destination', 'passanger', 'weather', 'coupon', 'expiration',
                'gender', 'maritalStatus', 'occupation','car',
            ]

            # ============================
            # 🟣 BINARY CATEGORICAL COLUMNS
            # ============================
            binary_columns = [
                'direction_same', 'direction_opp', 'toCoupon_GEQ5min',
                'toCoupon_GEQ15min', 'toCoupon_GEQ25min', 'has_children'
            ]

            # ============================
            # ⚙️ PIPELINES
            # ============================

            # Numerical pipeline
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            # Ordinal pipeline
            ordinal_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ordinal_encoder', OrdinalEncoder(categories=ordinal_mappings)),
                ('scaler', StandardScaler())
            ])

            # Nominal pipeline (OneHot)
            nominal_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore')),
                ('scaler', StandardScaler(with_mean=False))
            ])

            # Binary pipeline
            binary_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                # Convert True/False or Yes/No to 0/1
                ('ordinal_encoder', OrdinalEncoder()),
                ('scaler', StandardScaler())
            ])

            # ============================
            # 🔗 Combine all into ColumnTransformer
            # ============================
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, numerical_columns),
                    ('ordinal_pipeline', ordinal_pipeline, ordinal_columns),
                    ('nominal_pipeline', nominal_pipeline, nominal_columns),
                    ('binary_pipeline', binary_pipeline, binary_columns),
                ],
                remainder='drop'  # drop unused columns if any
            )

            logging.info("Preprocessor object created successfully")
            return preprocessor

        except Exception as e:
            logging.info("Exception occurred in data transformation setup")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessor object")

            preprocessing_obj = self.get_data_transformer_object()

            # Define target column
            target_column_name = 'Accept(Y/N?)'

            # Input features
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing datasets")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saving preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_path,
                obj=preprocessing_obj
            )

            logging.info("Data Transformation completed successfully")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
            pass

