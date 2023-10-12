#!/usr/bin/env python3
# ols.py                                                     SSimmons March 2018
"""
Uses a neural net to find the ordinary least-squares regression model. Trains
with batch gradient descent, and computes r^2 to gauge predictive quality.
"""

import torch
import pandas as pd
import torch.nn as nn
from itertools import combinations

# Read the named columns from the csv file into a dataframe.
names = ['taxvaluedollarcnt','finishedsquarefeet12','lotsizesquarefeet','yearbuilt','landtaxvaluedollarcnt',
         'fullbathcnt','roomcnt','finishedsquarefeet50']
         #'landtaxvaluedollarcnt','lotsizesquarefeet',
         #'fullbathcnt','calculatedfinishedsquarefeet','yearbuilt','bedroomcnt']
        # 'landtaxvaluedollarcnt'
        # 'fips'
        # 'finishedsquarefeet12','lotsizesquarefeet',
        # 'fips','fullbathcnt','yearbuilt', 'calculatedfinishedsquarefeet',
        # 'bedroomcnt', 'buildingqualitytypeid']
# building quality is a maybe
# choose square feet
# if we need more data then load other files as well
# goal 10-20 thousand
# get a sensible data set

#,'calculatedbathnbr','taxamount',
#    'buildingqualitytypeid','yearbuilt','assessmentyear','finishedsquarefeet6','calculatedfinishedsquarefeet',
#    'regionidneighborhood','bedroomcnt','basementsqft','lotsizesquarefeet']

# put zeros if not exist in pandas import
df = pd.read_csv('properties_2016.csv', low_memory=False,usecols=names)
print("OG Count")
print(df.count)
df.dropna(axis =  0, how='any', subset=None, inplace=True)
print("New count")
print(df.count)
#usecols = names, low_memory=False)
#non_null_counts = df.groupby(list(df.columns)).size()
#threshold=1000
#valid_combos = non_null_counts[non_null_counts>=threshold].index
#filtered_df = df.groupby(list(df.columns)).filter(lambda x: x.index in valid_combos)
#print(filtered_df.count)
#df.dropna(axis =  0, how='any', subset=None, inplace=True)
#print(filtered_df.count)
#quit()
data = df.values # read data into a numpy array (as a list of lists)
data = data[1:] # remove the first list which consists of the labels
data = data.astype(float) # coerce the entries in the numpy array to floats
data = torch.FloatTensor(data) # convert data to a Torch tensor
data.sub_(data.mean(0)) # mean-center
data.div_(data.std(0))
xss = data[:,1:]
yss = data[:,:1]

    

# define a model class
class LinearModel(nn.Module):
    # add a couple lines to make it non-linear
    # I'll be able to drive this up to higher than
    # risk of overfitting
    # linear models cannot overfit
    # check for overfiting by training 80% to a 100%
    # then for the rest of the data see if it drops down
    # if it does then it's non-linear
  def __init__(self):
    super(LinearModel, self).__init__()
    self.layer1 = nn.Linear(7, 1)

  def forward(self, xss):
    return self.layer1(xss)

# create and print an instance of the model class
model = LinearModel()
print(model)
z_parameters = []
for param in model.parameters():
    z_parameters.append(param.data.clone())
for param in z_parameters:
    param.zero_()

criterion = nn.MSELoss()

num_examples = len(data)
batch_size = 100
learning_rate = 0.009
epochs = 50
momentum = .99
# add momentum
# train the model

for epoch in range(epochs):
    indicies = torch.randperm(num_examples)
    accum_loss = 0
    for i in range(0, num_examples,batch_size):#num_examples//batch_size):
    # randomly pick batchsize examples from data
    # this code has replacement
    # best to do it without replacement
    # look at the last assignment and add momentum
        batch_indicies = indicies[i:i+batch_size]
    #indices = torch.randperm(num_examples)[:batch_size]

        yss_mb = yss[batch_indicies]#indicies]  # the targets for the mb (minibatch)
        yhatss_mb = model(xss[batch_indicies])#indices])  # model outputs for the mb

        loss = criterion(yhatss_mb, yss_mb)
        accum_loss+=loss.item()
        model.zero_grad()
        loss.backward() # back-propagate

    # update weights
        for i, (z_param, param) in enumerate(zip(z_parameters, model.parameters())):
              z_parameters[i] = momentum * z_param + param.grad.data
              param.data.sub_(z_parameters[i] * learning_rate)
    print("epoch: {0}, current loss: {1}".format(epoch+1, accum_loss * batch_size/num_examples))
        
    #print('epoch: {0}, loss: {1:11.8f}'.format(epoch+1, total_loss))
    with torch.no_grad():
        total_loss = criterion(model(xss), yss).item()
    
  #with torch.no_grad():
    #total_loss = criterion(model(xss), yss).item()
  

print("total number of examples:", num_examples, end='; ')
print("batch size:", batch_size)
print("learning rate:", learning_rate)

# Compute 1-SSE/SST which is the proportion of the variance in the data
# explained by the regression hyperplane.
SS_E = 0.0;  SS_T = 0.0
mean = yss.mean()
for xs, ys in zip(xss, yss):
  SS_E = SS_E + (ys - model(xs))**2
  SS_T = SS_T + (ys - mean)**2
print(f"1-SSE/SST = {1.0-(SS_E/SS_T).item():1.4f}")
