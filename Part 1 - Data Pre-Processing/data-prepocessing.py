#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
import librairies

@author: laurent
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import the dataset
dataset = pd.read_csv('Data.csv')

#Création de la matrice de fonctionnalités
X = dataset.iloc[:,:-1].values
#Création du vecteur dépendant
y = dataset.iloc[:, 3].values

#Taking care of missing data
from sklearn.preprocessing import Imputer
#Fait la moyenne pour remplacer les données manquantes
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
#Transform remplace les données avec la moyenne
X[:, 1:3] = imputer.transform(X[:, 1:3])

#Encoding categorical data
#Permet de remplacer les mots par des 0,1,2..
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#Country
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
onehotencoder = OneHotEncoder(categorical_features=[0])
X=onehotencoder.fit_transform(X).toarray()
#Purchased
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#Splitting the dataset into train and test
from sklearn.cross_validation import train_test_split
#divise les données entre 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#features scaling -->centrées les données
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)   
