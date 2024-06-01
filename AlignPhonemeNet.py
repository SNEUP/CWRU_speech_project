import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as P
import torch.nn.utils.parametrizations as PT
from utils import Sphere,CustomDataset,CustomDataLoader
from torch.utils.data import DataLoader

def cost_func_confusion_matrix(X:torch.Tensor, y:torch.Tensor):
    '''
    This cost function compute the mean of element-wise L2 distance of two similarity matrices.
    :param X: low D embedding matrix. (n_sample x n_feature)
    :param y: the reference similarity matrix. (n_sample x n_sample)
    :param sim_metrics: similarity metrics ('cos' or 'corr')
    :return: mean L2 distance
    '''
    assert X.size(0) == y.size(0)

    # normalize X
    X=X/X.norm(dim=-1, keepdim=True)

    # compute the similarity matrix
    similarity_matrix = torch.mm(X, X.t())

    # compute the sum of element-wise L2 distance
    distance=torch.sum((similarity_matrix-y)**2)

    # return the mean of the distance
    return distance/X.size(0)**2

class AlignPhonemeNet(nn.Module):
    def __init__(self, n_features, n_low_D):
        super(AlignPhonemeNet, self).__init__()
        self.n_low_D = n_low_D
        self.n_features = n_features

        # define the linear projection matrix: (1) no bias (2) sum of the squared weights is 1 for each low D dimesion
        self.linear_X = P.register_parametrization(nn.Linear(n_features, n_low_D,bias=False), 'weight', Sphere(dim=-1))
        self.tanh=nn.Tanh()

        # save the losses and weights
        self.training_loss=[]
        self.validation_loss=[]
        self.model_weights=[]
        self.best_weights=None
        self.best_training_loss=None

    def refresh(self):
        # save the losses and weights
        self.training_loss=[]
        self.validation_loss=[]
        self.model_weights=[]
        self.best_weights=None
        self.best_training_loss=None
        nn.init.normal_(self.linear_X.weight, mean=0,std=0.5) # some random initialization

    def forward(self, X):
        X_embed=self.tanh(self.linear_X(X))
        return X_embed

    def fit(self, X:np.ndarray,Y:np.ndarray,
            X_val=None, Y_val=None,
            num_iterations=500, learning_rate=0.01, lambda_l1=0.001,device='cpu'):
        '''
        :param X: the high dimensional data. (n_sample x n_features)
        :param Y: the corresponding similarity matrix. (n_sample x n_sample)
        :param X_val: the high dimensional data. (n_sample_val x n_features)
        :param Y_val: the correponding similarity matrix. (n_sample_val x n_sample_val)
        :param num_iterations: stopping iteration: int
        :param learning_rate: default:0.01
        :param lambda_l1: the L1 penalty of the linear matrix
        :param device: default cpu. But if run on GPU ('cuda') it's a lot faster.
        :return:
        '''
        self.refresh() # reset everything before fitting
        self.to(device)
        X = torch.from_numpy(X).float().to(device)
        Y = torch.from_numpy(Y).float().to(device)

        if X_val is not None:
            assert isinstance(X_val, np.ndarray) and isinstance(Y_val, np.ndarray)
            X_val=torch.from_numpy(X_val).float().to(device)
            Y_val=torch.from_numpy(Y_val).float().to(device)

        # define a generic optimizer
        optimizer=torch.optim.Adam(self.parameters(), lr=learning_rate)

        # training process
        for _ in range(num_iterations):
            # generate the low D embedding of neural data
            X_embed = self.forward(X)

            # calculate the loss: similarity loss + l1 penalty
            loss=cost_func_confusion_matrix(X_embed,Y)
            l1_reg = lambda_l1 * torch.norm(self.linear_X.weight, p=1)
            loss += l1_reg
            self.training_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if there is a validation set. This is to determine if the model is overfitting
            if X_val is not None:
                with torch.no_grad():
                    X_test_embed= self.forward(X_val)
                    loss_test=cost_func_confusion_matrix(X_test_embed,Y_val)
                    l1_reg = lambda_l1 * torch.norm(self.linear_X.weight, p=1)
                    loss_test += l1_reg
                    self.validation_loss.append(loss_test.item())
            self.model_weights.append(self.state_dict())

        # find the optimal model
        if X_val is not None:
            self.best_weights=self.model_weights[np.argmin(self.validation_loss)]
            self.best_training_loss = self.training_loss[np.argmin(self.validation_loss)]
        else:
            self.best_weights=self.model_weights[-1]
            self.best_training_loss = self.training_loss[-1]

    def fit_minibatch(self, X:np.ndarray,Y:np.ndarray,
            X_val, Y_val,
            n_epochs=500, batch_size=10,patience=50,
            learning_rate=0.01, lambda_l1=0.001,device='cpu'):
        '''
        :param X: the high dimensional data. (n_sample x n_features)
        :param Y: the corresponding similarity matrix. (n_sample x n_sample)
        :param X_val: the high dimensional data. (n_sample_val x n_features). Mandatory argument
        :param Y_val: the correponding similarity matrix. (n_sample_val x n_sample_val) Mandatory argument
        :param num_iterations: stopping iteration: int
        :param learning_rate: default:0.01
        :param batch_size: the minibatch size
        :param patience: the minibatch patience.
                If patience is 0, the model will train all n_epochs.
                Otherwise it will wait number of patience epochs to see if the validation loss decrease again.
                If the validation loss is noisy. This value should be larger.
        :param lambda_l1: the L1 penalty of the linear matrix. We would like this matrix to be sparse.
        :param device: default cpu. But if run on GPU ('cuda') it's a lot faster.
        :return:
        '''
        self.refresh() # reset everything before fitting
        self.to(device)
        # prepare training set
        X = torch.from_numpy(X).float().to(device)
        Y = torch.from_numpy(Y).float().to(device)
        train_dataset=CustomDataset(X,Y)

        # prepare validation set: No minibatch on the validation set
        X_val=torch.from_numpy(X_val).float().to(device)
        Y_val=torch.from_numpy(Y_val).float().to(device)


        # define a generic optimizer
        optimizer=torch.optim.Adam(self.parameters(), lr=learning_rate)

        # set the best loss
        best_loss=float('inf')
        counter=0

        # training process
        for epoch in range(n_epochs):
            train_loader = CustomDataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
            # training with mini-batch
            train_loss = 0.0
            for batch in iter(train_loader):
                X_mini_batch,y_mini_batch=batch
                X_embed = self.forward(X_mini_batch)

                # calculate the loss: similarity loss + l1 penalty
                loss=cost_func_confusion_matrix(X_embed,y_mini_batch)
                l1_reg = lambda_l1 * torch.norm(self.linear_X.weight, p=1)
                loss += l1_reg
                train_loss+=loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # record the training loss
            train_loss/=len(train_loader)
            self.training_loss.append(train_loss)

            with torch.no_grad():
                X_val_embed= self.forward(X_val)
                loss_val=cost_func_confusion_matrix(X_val_embed,Y_val)
                l1_reg = lambda_l1 * torch.norm(self.linear_X.weight, p=1)
                loss_val += l1_reg
                val_loss=loss_val.item()
                self.validation_loss.append(val_loss)
            self.model_weights.append(self.state_dict())
            print(f"Epoch {epoch + 1}/{n_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            if patience!=0:
                if val_loss < best_loss:
                    best_loss = val_loss
                    counter = 0
                else:
                    counter += 1
                    if counter >= patience:
                        print(f"Early stopping at epoch {epoch + 1}")
                        self.best_training_loss=best_loss
                        self.best_weights=self.model_weights[-counter]
                        break
        # if patience is 0 we are running the entire n_epoch and then find the minimum val loss.
        if patience==0:
            self.best_weights=self.model_weights[np.argmin(self.validation_loss)]
            self.best_training_loss = self.training_loss[np.argmin(self.validation_loss)]


    def predict(self, X_test, device='cuda'):
        assert self.best_weights is not None, "Please fit before predicting."
        self.to(device)
        X_test = torch.from_numpy(X_test).float().to(device)
        with torch.no_grad():
            self.load_state_dict(self.best_weights)
            X_embed = self.forward(X_test)
        return X_embed

    def score(self,X_test,Y_test,device='cuda'):
        assert self.best_weights is not None, "Please fit before scoring."
        self.weights = self.best_weights
        self.to(device)
        X_test=torch.from_numpy(X_test).float().to(device)
        Y_test=torch.from_numpy(Y_test).float().to(device)
        with torch.no_grad():
            self.load_state_dict(self.best_weights)
            X_embed= self.forward(X_test)
            loss = cost_func_confusion_matrix(X_embed, Y_test)
        return loss.cpu().detach().numpy()


























