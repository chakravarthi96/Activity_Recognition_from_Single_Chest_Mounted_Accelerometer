# Activity Recognition from Single Chest Mounted Accelerometer

Modelling uncalibrated accelerometer data are collected from 15 participantes performing 7 activities. The dataset provides challenges for identification and authentication of people using motion patterns.

# Setup
To setup the whole project just clone repo and run the following command

    bash setup.sh
    
After this logs are available from

    tail -f service.log
    
Service is up and running at `http://127.0.0.1:8000/`

# Files

**HELPER FOLDER** contains the following python files.

- ingest.py
- process.py
- improve.py

**ingest.py** contains the code to collate all the .csv which are part of the dataset into dataframe.\
**process.py** contains the code to perform modelling against the dataset with different algorithms\
**improve.py** contains the code to perform feature selection and improvement of accuracy defined by the scope of the algorithm\

# Requirements

- fastapi==0.54.2
- numpy==1.18.4
- pandas==1.0.3
- pydantic==1.5.1
- scikit-learn==0.22.2.post1
- sklearn==0.0
- starlette==0.13.2
- tqdm==4.52.0
- uvicorn==0.11.5
- uvloop==0.14.0

# Algorithms' Accuracy Report

- **Logistic Regression** _(64.33)_
- **K Neighbors** _(77.79)_
- **Bernoulli NB** _(51.87)_
- **Complement NB** _(62.96)_
- **Gaussian NB** _(72.82)_
- **Multinomial NB** _(56.61)_
- **Linear Support Vector** _(51.72)_
- **Support Vector** _(75.38)_
- **Decision Tree** _(69.94)_
- **Extra Tree** _(75.20)_
- **Multi Layer Perceptron** _(52.48)_
- **Ada Boost** _(76.57)_
- **Bagging** _(75.26)_
- **Extra Trees** _(75.58)_
- **Histogram Based Gradient Boosting** _(51.87)_
- **Random Forest** _(68.41)_
- **Voting Classifier** _(72.73)_

# Reason why KNN was choosen as the Algorithm to be improved upon
- The weight's file is lighter than the others listed in the Accuracy Report.
- The time taken to train is lesser than the others.
- The inference time is faster than the others.

# Improving KNN

**KNN** with following features

- **Neighbours** is _38_
- **Weight Type** is _uniform_
- **Metric** is _minkowski_
- **Leaf Size** is _10_

**Accuracy** achieved _78.81_

# API

To start the server

    uvicorn app:app --reload

Visit `http://127.0.0.1:8000/`
To use it as micro-service

Example

- **X** = _1500_
- **Y** = _1500_
- **Z** = _1500_

```bash
curl -X GET "http://127.0.0.1:8000/api/v1/predict?X=1500&Y=1500&Z=1500" -H  "accept: application/json"
```
