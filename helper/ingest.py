import pandas as pd
from os import listdir
from os.path import isfile, join


def load_data(directory):

    files = [f for f in listdir(directory) if isfile(join(directory, f))]
    for index, file in enumerate(files):
        if index == 0:
            response = pd.read_csv(directory + file)
        else:
            response.append(pd.read_csv(directory + file), ignore_index=True)
    response.columns = ["Sequence", "X-Axis", "Y-Axis", "Z-Axis", "Activity"]
    response = response.drop(["Sequence"], axis=1)
    response = response.sample(frac=1).reset_index(drop=True)

    return response