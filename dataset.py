import numpy as np
import csv
from sklearn.utils import Bunch

def load_cardio_dataset():
    with open(r'Cardiovascular Disease Prediction using Machine Learing/cardio_train.csv') as csv_file:
        data_reader = csv.reader(csv_file, delimiter=";")
        feature_names = next(data_reader)[:-1]
        data = []
        target = []
        for row in data_reader:
            features = row[:-1]
            label = row[-1]
            data.append([float(num) for num in features])
            target.append(int(label))
        
        data = np.array(data)
        target = np.array(target)
    return Bunch(data=data, target=target, feature_names=feature_names)