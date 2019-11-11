
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import json
import psycopg2
import datetime as dt
import dateutil.parser
import re
import csv
from os import listdir
from functools import reduce
from scipy import stats
from decimal import Decimal
from IPython.core.interactiveshell import InteractiveShell
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('dark_background')
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# database variables
# cred     = json.load(open('credential.json'))
# db1 = 'gemini'
# db2 = 'engine'
# port     = '55432'
# user1     = cred[db1]['username']
# password1 = cred[db1]['password']
# user2     = cred[db2]['username']
# password2 = cred[db2]['password']
# localhost = '127.0.0.1'
# conn_gem = psycopg2.connect(host=localhost,
#     port=port,
#     user=user1,
#     password=password1,
#     dbname=db1)
# conn_eng = psycopg2.connect(host=localhost,
#     user=user2,
#     password=password2,
#     dbname=db2)
# def query(sql, c):
#     return pd.read_sql_query(sql, c)
# def query_file(sql_file, c):
#     with open(sql_file, "r") as f:
#         sql = f.read()
#     return query(sql, c)

pd.options.display.max_colwidth = 100
pd.set_option('display.max_columns', 100)
# db2 = 'notus'
# conn_notus = psycopg2.connect(host=localhost,

def query(sql, c):
	return pd.read_sql_query(sql, c)
def query_file(sql_file, c):
	with open(sql_file, "r") as f:
		sql = f.read()
	return query(sql, c)
import requests
from dateutil.relativedelta import relativedelta
from nameparser import HumanName
from ast import literal_eval
cred     = json.load(open('credential.json'))
database = 'gemini'
port     = '55432'
user     = cred[database]['username']
password = cred[database]['password']
localhost = '127.0.0.1'
def query(sql):
	conn = psycopg2.connect(host=localhost,
		port=port,
		user=user,
		password=password,
		dbname=database)
	return pd.read_sql_query(sql, conn)
def query_file(sql_file):
	return query(sql)
import maxminddb
from geopy import distance
# Machine Learning Imports
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
plt.style.use('bmh')
sns.set_style("dark")
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 1000)
# For evaluating our ML results
from sklearn import metrics
db1 = 'gemini'
db2 = 'engine'
user1     = cred[db1]['username']
password1 = cred[db1]['password']
user2     = cred[db2]['username']
password2 = cred[db2]['password']
conn_gem = psycopg2.connect(host=localhost,
	port=port,
	user=user1,
	password=password1,
	dbname=db1)
conn_eng = psycopg2.connect(host=localhost,
	user=user2,
	password=password2,
	dbname=db2)
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import RFECV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
# from spark_sklearn import GridSearchCV
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import roc_curve, roc_auc_score, classification_report
pd.set_option('display.max_columns', 150)
