import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
import dill

def save_numpy_array_data(file_path, obj):
    """Save a numpy array to a file using dill.

    Args:
        file_path (str): The path to the file where the array will be saved.
        obj (np.ndarray): The numpy array to save.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)