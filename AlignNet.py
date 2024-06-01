import torch
import torch.nn as nn
import torch.nn.utils.parametrize as P
import torch.nn.utils.parametrizations as PT
from tqdm import tqdm
from sklearn.model_selection import train_test_split, KFold
from scipy.ndimage.filters import gaussian_filter
import numpy as np
from utils import select_submatrix,Sphere

# this cost function calculate the correlation coefficient for each feature (not for each sample)
def cost_func_dep(x:torch.Tensor,y:torch.Tensor):
    X_centered = x - torch.mean(x, dim=0)[None, :]
    Y_centered = y - torch.mean(y, dim=0)[None, :]
    covariance = torch.diag(torch.matmul(X_centered.T, Y_centered))
    X_std = torch.sqrt(torch.sum(X_centered ** 2, dim=0)) + 10e-5
    Y_std = torch.sqrt(torch.sum(Y_centered ** 2, dim=0)) + 10e-5
    correlation = covariance / (X_std * Y_std)
    loss = 1 - torch.mean(correlation)
    return loss
def cost_func(X:torch.Tensor,y:torch.Tensor):
    all=torch.concat([X,y],dim=0)
    all_mean=all.mean()
    all_std=all.std()
    X=(X-all_mean)/all_std
    y=(y-all_mean)/all_std
    distance=torch.mean((X - y) ** 2)

    return distance#/(torch.std(X)*torch.std(y))
def cost_func_confusion_matrix(X:torch.Tensor,y:torch.Tensor):
    # here X is the regular data matrx
    # Y is a confusion matrix
    # Normalize the data matrix
    norms = X.norm(dim=1, keepdim=True)
    normalized_tensor = X / norms

    # Compute the cosine similarity matrix using matrix multiplication
    similarity_matrix = torch.mm(normalized_tensor, normalized_tensor.t())
    # Compute the correlation matrix
    #similarity_matrix=torch.cov(normalized_tensor)/torch.var(normalized_tensor)

    return torch.sum((similarity_matrix-y)**2)/similarity_matrix.size(0)**2

class AlignmentNet(nn.Module):
    def __init__(self, n_features, n_targets, n_low_D):
        super(AlignmentNet, self).__init__()
        self.n_low_D = n_low_D
        self.linear_X = P.register_parametrization(nn.Linear(n_features, n_low_D,bias=False), 'weight', Sphere(dim=-1))
        self.linear_Y = P.register_parametrization(PT.orthogonal(nn.Linear(n_targets, n_low_D,bias=False)), 'weight', Sphere(dim=-1))
        self.tanh = nn.Tanh()
        self.training_loss=[]
        self.validation_loss=[]
        self.model_weights=[]
        self.best_weights=None
        self.best_training_loss=None
        self.linear_X_weights=[]
        self.linear_Y_weights=[]

    def forward(self, X, Y):
        X_embed = self.tanh(self.linear_X(X))
        Y_embed = self.tanh(self.linear_Y(Y))
        return X_embed, Y_embed

    def fit(self, X, Y, X_val=None,Y_val=None,num_iterations=500, learning_rate=0.001, lambda_l1=0.0001, device='cuda',test_size=0.2,fix_y=False):
        self.to(device)
        self.training_loss=[]
        self.validation_loss=[]
        self.model_weights=[]
        self.best_weights=None
        X=torch.from_numpy(X).float().to(device)
        Y=torch.from_numpy(Y).float().to(device)
        if fix_y:
            for param in self.linear_Y.parameters():
                param.requires_grad=False
        else:
            for param in self.linear_Y.parameters():
                param.requires_grad = True
        if X_val is not None:
            X_val=torch.from_numpy(X_val).float().to(device)
            Y_val=torch.from_numpy(Y_val).float().to(device)

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        for _ in range(num_iterations):
            X_embed, Y_embed = self.forward(X, Y)

            loss=cost_func(X_embed,Y_embed)
            l1_reg = lambda_l1 * torch.norm(self.linear_X.weight, p=1)
            loss += l1_reg
            self.training_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # validation set
            if X_val is not None:
                with torch.no_grad():
                    X_test_embed, Y_test_embed = self.forward(X_val, Y_val)
                    loss_test=cost_func(X_test_embed,Y_test_embed)
                    l1_reg = lambda_l1 * torch.norm(self.linear_X.weight, p=1)
                    loss_test += l1_reg
                    self.validation_loss.append(loss_test.item())

            self.model_weights.append(self.state_dict())
        if X_val is not None:
            self.best_weights=self.model_weights[np.argmin(self.validation_loss)]
            self.best_training_loss = self.training_loss[np.argmin(self.validation_loss)]

    def _orthonormal_constraint(self):
        with torch.no_grad():
            U, _, Vt = torch.svd(self.linear_Y.weight)
            self.linear_Y.weight.data = torch.mm(U, Vt.T)

    def predict(self, X_test, Y_test, device='cuda'):
        assert self.best_weights is not None, "Please fit before predicting."

        self.to(device)
        X_test=torch.from_numpy(X_test).float().to(device)
        Y_test=torch.from_numpy(Y_test).float().to(device)
        with torch.no_grad():
            self.load_state_dict(self.best_weights)
            X_embed, Y_embed = self.forward(X_test, Y_test)
        return X_embed, Y_embed

    def score_deprecated(self, X_test, Y_test, device='cuda'):
        assert self.best_weights is not None, "Please fit before scoring."
        self.weights = self.best_weights
        self.to(device)
        X_test=torch.from_numpy(X_test).float().to(device)
        Y_test=torch.from_numpy(Y_test).float().to(device)
        with torch.no_grad():
            self.load_state_dict(self.best_weights)
            X_embed, Y_embed = self.forward(X_test, Y_test)
            ss_res = torch.sum((Y_embed - X_embed) ** 2, dim=1)
            ss_tot = torch.sum((Y_embed - torch.mean(Y_embed, dim=0)) ** 2, dim=1)
            r2 = 1 - torch.mean(ss_res / ss_tot)
        return r2.item()

    def score(self,X_test,Y_test,device='cuda'):
        assert self.best_weights is not None, "Please fit before scoring."
        self.weights = self.best_weights
        self.to(device)
        X_test=torch.from_numpy(X_test).float().to(device)
        Y_test=torch.from_numpy(Y_test).float().to(device)
        with torch.no_grad():
            self.load_state_dict(self.best_weights)
            X_embed, Y_embed = self.forward(X_test, Y_test)
            loss = cost_func(X_embed, Y_embed)
        return loss.cpu().detach().numpy()

    def assign_Y_weights(self,weights,device='cuda'):
        # weights can be tensor or numpy array
        # this will assign the weights and freeze the weights of linear Y weights
        if isinstance(weights,np.ndarray):
            weights = torch.from_numpy(weights).float().to(device)
        else:
            assert isinstance(weights,torch.Tensor)
        with torch.no_grad():
            self.linear_Y.weight.copy_(weights)
        for param in self.linear_Y.parameters():
            param.requires_grad = False


class AlignPhonemeNet(nn.Module):
    def __init__(self, n_features, n_low_D):
        super(AlignPhonemeNet, self).__init__()
        self.n_low_D = n_low_D
        self.linear_X = P.register_parametrization(nn.Linear(n_features, n_low_D,bias=False), 'weight', Sphere(dim=-1))
        #self.linear_X=nn.Linear(n_features, n_low_D,bias=False)
        self.tanh = nn.Tanh()
        self.training_loss=[]
        self.validation_loss=[]
        self.model_weights=[]
        self.best_weights=None
        self.best_training_loss=None

    def forward(self, X):
        X_embed = self.tanh(self.linear_X(X))
        return X_embed

    def fit(self, X, Y, X_val=None,Y_val=None,num_iterations=500, learning_rate=0.001, lambda_l1=0.0001, device='cuda',test_size=0.2):
        self.to(device)
        self.training_loss=[]
        self.validation_loss=[]
        self.model_weights=[]
        self.best_weights=None
        X=torch.from_numpy(X).float().to(device)
        Y=torch.from_numpy(Y).float().to(device) # here Y should be the distance matrix of the whole word phonemes
        if X_val is not None:
            X_val=torch.from_numpy(X_val).float().to(device)
            Y_val=torch.from_numpy(Y_val).float().to(device)

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        for _ in range(num_iterations):
            X_embed = self.forward(X)

            loss=cost_func_confusion_matrix(X_embed,Y)
            l1_reg = lambda_l1 * torch.norm(self.linear_X.weight, p=1)
            loss += l1_reg
            self.training_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # validation set
            if X_val is not None:
                with torch.no_grad():
                    X_test_embed= self.forward(X_val)
                    loss_test=cost_func_confusion_matrix(X_test_embed,Y_val)
                    l1_reg = lambda_l1 * torch.norm(self.linear_X.weight, p=1)
                    loss_test += l1_reg
                    self.validation_loss.append(loss_test.item())
            self.model_weights.append(self.state_dict())
        if X_val is not None:
            self.best_weights=self.model_weights[np.argmin(self.validation_loss)]
            self.best_training_loss = self.training_loss[np.argmin(self.validation_loss)]

    def predict(self, X_test, device='cuda'):
        assert self.best_weights is not None, "Please fit before predicting."

        self.to(device)
        X_test=torch.from_numpy(X_test).float().to(device)
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


if __name__ == '__main__':
    from Bolu_IFG.utils import *
    from sklearn.decomposition import PCA
    import enlighten
    import matplotlib.pyplot as plt

    data_to_use=np.load('../Bolu_IFG/data_to_use.npy')
    embedding_to_use=np.load('../Bolu_IFG/embedding_to_use.npy')

    # for fixing the word embedding
    n_components=10
    embedding_to_use=PCA(n_components=n_components).fit_transform(embedding_to_use)

    # Here we are trying to do some regression task. It's shown that the dataset m
    preprocess_dim = 40

    ### preprocesing
    preprocessor = PCA(n_components=preprocess_dim)

    # define how many bins we want in each feature
    n_bins_per_feature = 10

    # reformat the data: for each time bin we include n_bins_per_feature of history
    X = reformat(data_to_use, n_bins_per_feature)  # data is a list

    #### Start decoding
    n_repeat = 5
    averaged_score = []
    averaged_chance = []
    # GUI for process visualization
    manager = enlighten.get_manager()
    ticks = manager.counter(total=n_repeat, desc="Num. repeat", unit="repeats", color="red")

    for j in range(n_repeat):
        score = []
        chance = []
        for n_bin in [47,48,49]:
            # sample the train and test set
            X_train, X_test, y_train, y_test,_,_ = get_train_test(X[n_bin], embedding_to_use, train=0.8)

            # decrease the dimension
            preprocessor.fit(X_train)
            X_train_low_D = preprocessor.transform(X_train)
            X_test_low_D = preprocessor.transform(X_test)

            extractor = AlignmentNet(n_features=X_train_low_D.shape[1],
                                     n_targets=y_train.shape[1], n_low_D=5)
            #extractor.assign_Y_weights(np.eye(n_components),'cuda')
            X_train_low_D = (X_train_low_D - np.mean(X_train_low_D)) / np.std(X_train_low_D)
            X_test_low_D = (X_test_low_D - np.mean(X_test_low_D)) / np.std(X_test_low_D)
            y_train = (y_train - np.mean(y_train)) / np.std(y_train)
            y_test = (y_test - np.mean(y_test)) / np.std(y_test)
            # do the regression(alignment)
            extractor.fit(X_train_low_D, y_train, learning_rate=0.01,lambda_l1=0.0005, num_iterations=500,fix_y=False)
            sc = extractor.score(X_test_low_D, y_test)

            np.random.shuffle(X_train_low_D.T)
            np.random.shuffle(y_train.T)
            extractor.fit(X_train_low_D, y_train, learning_rate=0.01, lambda_l1=0.0005, num_iterations=500,fix_y=False)
            ch = extractor.score(X_test_low_D, y_test)

            score.append(sc)
            chance.append(ch)
        ticks.update()
        averaged_score.append(score)
        averaged_chance.append(chance)





























