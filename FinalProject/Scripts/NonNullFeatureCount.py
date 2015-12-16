import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from mpl_toolkits.mplot3d import Axes3D
import csv

with open("attributeCounts.csv",'wr+') as f:
	f.write("Name,Column Index,Count\n")
	baseFileName = "../Data/MERGED"
	for i in range(2013,2014):
		tempFileName=baseFileName+str(i)+"_PP.csv"
		with open(tempFileName,mode='rb') as file:
		    newData = pd.DataFrame(pd.read_csv(file))
		    for j in range(0,(len(newData.columns))):
				nonNullData = (newData.loc[:,newData.columns[j]]).dropna()
				nonNullData = pd.DataFrame(nonNullData)
				if nonNullData[nonNullData.columns[0]].dtype  == "object":
				    nonNullNotSuppressed = nonNullData[nonNullData[nonNullData.columns[0]].str.contains("PrivacySuppressed") == False]
				else:
					nonNullNotSuppressed = nonNullData
				if nonNullNotSuppressed.size > 0:
				    f.write(str(nonNullNotSuppressed.columns[0])+","+str(j)+","+str(nonNullNotSuppressed.size)+'\n')
				
