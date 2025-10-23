import os 
import sys
from src.MLproject.exception import CustomException
from src.MLproject.logger import logging
import pandas as pd


from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str= os.path.join('artifacts','train.csv')
    test_data_path: str= os.path.join('artifacts','test.csv')
    raw_data_path: str= os.path.join('artifacts','raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config= DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion method starts")
        try:
            df= pd.read_csv('notebook/data/stud.csv')
            logging.info("reading from sql database")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False)
            logging.info("Raw data is saved")

            from sklearn.model_selection import train_test_split

            train_set, test_set= train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)