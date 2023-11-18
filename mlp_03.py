import numpy as np
import pandas as pd
import logging
import anndata as ad
import pickle

logging.basicConfig(level=logging.INFO)

from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt

import logging
import anndata as ad
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from scipy.sparse import csc_matrix

from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import scale

logging.basicConfig(level=logging.INFO)

train_gex = ad.read_h5ad("../Gex_processed_training.h5ad") #gex is gene expression which are RNA; Training data input
train_adt = ad.read_h5ad("../Adt_processed_training.h5ad") # adt is protein; Training data response

test_gex = ad.read_h5ad("../Gex_processed_testing.h5ad") #gex is gene expression which are RNA; Training data input
test_adt = ad.read_h5ad("../Adt_processed_testing.h5ad") # adt is protein; Training data response


# This will get passed to the method
input_train_gex = train_gex
input_train_adt = train_adt
input_test_gex =  test_gex

# This will get passed to the metric
true_test_adt =  test_adt

def calculate_rmse(true_test_adt, pred_test_adt):
    return  mean_squared_error(true_test_adt.X.toarray(), pred_test_adt.X, squared = False)

def baseline_linear(input_train_gex, input_train_adt, input_test_gex):
    '''Baseline method training a linear regressor on the input data'''
    input_gex = ad.concat(
        {"train": input_train_gex, "test": input_test_gex},
        axis = 0,
        join = "outer",
        label = "group",
        fill_value = 0,
        index_unique = "-", 
    )
    
    # Do PCA on the input data
    n = 50
    logging.info('Performing dimensionality reduction on GEX values...')
    embedder_gex = TruncatedSVD(n_components=n)
    gex_pca = embedder_gex.fit_transform(input_gex.X)

    with open('pca_model.pkl', 'wb') as f:
        pickle.dump(embedder_gex, f)
    
    # split dimension reduction GEX back up for training
    X_train = gex_pca[input_gex.obs['group'] == 'train']
    X_test = gex_pca[input_gex.obs['group'] == 'test']
    y_train = input_train_adt.X.toarray()
    
    assert len(X_train) + len(X_test) == len(gex_pca)
    
    logging.info('Running Linear regression...')
    
    reg = Ridge()
    
    # Train the model on the PCA reduced gex 1 and 2 data   
    # Hyperparameters for our network
    input_size = n
    hidden_sizes = [5000,5000,5000]
    output_size = 134

    # Build a feed-forward network
    model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                          nn.BatchNorm1d(hidden_sizes[0]),
                          nn.ReLU(),
                          nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                          nn.BatchNorm1d(hidden_sizes[1]),
			  nn.ReLU(),
			  nn.Linear(hidden_sizes[1], hidden_sizes[2]),
                          nn.BatchNorm1d(hidden_sizes[2]),
                          nn.Tanh(),
                          nn.Linear(hidden_sizes[2], output_size))
    print(model)

    print("Building tensor objects")

    X_train_tensor = torch.Tensor(X_train)
    y_train_tensor = torch.Tensor(y_train)
    X_test_tensor  = torch.Tensor(X_test)
    #y_test_tensor  = torch.Tensor(y_test)

    mydata_train = TensorDataset(X_train_tensor,y_train_tensor)
    #mydata_test  = TensorDataset(X_test_tensor,y_test_tensor)

    trainloader = torch.utils.data.DataLoader(mydata_train, batch_size=20, shuffle=True, drop_last=True)
    #testloader = torch.utils.data.DataLoader(mydata_test, batch_size=10, shuffle=True, drop_last=True)

    criterion = nn.MSELoss()# Optimizers require the parameters to optimize and a learning rate
    optimizer = optim.SGD(model.parameters(), lr=0.003)
    epochs = 15
 
    print("Running epochs")

    for e in range(epochs):
        running_loss = 0
        for data, target in trainloader:

            # Training pass
            optimizer.zero_grad()

            output = model(data) #<--- note this line is using the model you set up at the beginning of this section
            output = output.float()
            target = target.float()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        else:
            print(f"Training loss: {running_loss/len(trainloader)}")

    predict = model(X_test_tensor)
    y_pred = predict.detach().numpy()

    # Save the model
    PATH = "model_01.pth"
    torch.save(model.state_dict(), PATH)
 
    # Project the predictions back to the adt feature space
    
    pred_test_adt = ad.AnnData(
        X = y_pred,
        obs = input_test_gex.obs,
        var = input_train_adt.var,
    
    )
    
    # Add the name of the method to the result
    pred_test_adt.uns["method"] = "linear"
    
    return pred_test_adt

def baseline_mean(input_train_gex, input_train_adt, input_test_gex):
    '''Dummy method that predicts mean(input_train_adt) for all cells'''
    logging.info('Calculate mean of the training data adt...')
    y_pred = np.repeat(input_train_adt.X.mean(axis = 0).reshape(-1, 1).T, input_test_gex.shape[0], axis=0)
    
    # Prepare the ouput data object
    pred_test_adt = ad.AnnData(
        X = y_pred,
        obs = input_test_gex.obs,
        var = input_train_adt.var,
    )
    
    pred_test_adt.uns["method"] = "mean"

    return pred_test_adt

for method in [baseline_linear]:
    # Run prediction
    pred_test_adt = method(input_train_gex, input_train_adt, input_test_gex)
    # Calculate RMSE
    rmse = calculate_rmse(true_test_adt, pred_test_adt)
    # Print results
    print(f'{pred_test_adt.uns["method"]} had a RMSE of {rmse:.4f}')
