from src.MLproject.exception import CustomException
from src.MLproject.logger import logging
import sys

if __name__=="__main__":
    logging.info("the execution has started")
     
    try:
        a=1/0
    except Exception as e:
        logging.info("caught an exception")
        raise CustomException(e,sys)