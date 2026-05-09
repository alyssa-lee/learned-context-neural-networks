import torch
from torch import nn
from torch.utils.data import Dataset

import numpy as np
import polars as pl
import matplotlib.pyplot as plt

def load_height_data(centered=False):
    df_orig = pl.read_csv("data/NCD_RisC_Lancet_2020_height_child_adolescent_country.csv")
    df = df_orig.to_dummies(columns="Sex")
    df = df.rename({"Sex_Boys": "Sex"})
    df = df.with_columns(pl.col("Year").sub(1985).alias("Year_since_1985"))
    df = df.select(["Country", "Sex", "Year_since_1985", "Age group", "Mean height"])
    df = df.select(
        pl.col("Country"),
        pl.col("Sex").cast(pl.Float32),
        pl.col("Year_since_1985").cast(pl.Float32),
        pl.col("Age group").cast(pl.Float32),
        pl.col("Mean height").cast(pl.Float32),
    )

    df_X = np.array(df.select(["Sex", "Year_since_1985", "Age group"]))
    df_Y = np.array(df.select("Mean height"))
    df_C = np.array(df.select("Country"))

    if centered:
        X_mu = np.mean(df_X, axis=0)
        Y_mu = np.mean(df_Y, axis=0)
        X_centered = df_X - X_mu
        Y_centered = df_Y - Y_mu
        return X_centered, Y_centered, df_C


    return df_X, df_Y, df_C


class HeightDataset(Dataset):
    def __init__(self):
        self.X, self.Y, self.C = load_height_data(centered=True)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.Y[idx])


class ContextSensitiveHeightDataset(Dataset):
    def __init__(self):
        X_data, self.Y, self.C = load_height_data(centered=True)
        self.context_labels = np.unique(self.C)
        C_df = pl.DataFrame({"Country": self.C.flatten()})
        C_df = C_df.to_dummies()
        C_df = C_df.cast({pl.selectors.numeric(): pl.Float32})
        C_one_hot = torch.from_numpy(np.array(C_df))
        self.X = torch.cat((torch.from_numpy(X_data), C_one_hot), dim=1)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], torch.from_numpy(self.Y[idx])


def load_brca_data(centered=False, context="her2_er"):
    df_orig = pl.read_csv("data/METABRIC_RNA.csv", infer_schema_length=1000)

    df_meta = df_orig[:,:31]
    df_X = np.array(df_orig[:,31:].cast({pl.selectors.numeric(): pl.Float32}))

    df_Y = np.array(df_meta.select(
        pl.col("overall_survival").cast(pl.Float32)
    ))
    
    df_C_all = df_meta.with_columns(
        pl.concat_str(
            [
                pl.col("her2_status"),
                pl.col("er_status")
            ],
            separator="_",
        ).alias("her2_er"),
        pl.concat_str(
            [
                pl.col("chemotherapy"),
                pl.col("radio_therapy"),
                pl.col("hormone_therapy")
            ],
            separator="_",
        ).alias("chemo_radio_hormone"),
        pl.concat_str(
            [
                pl.col("her2_status"),
                pl.col("er_status"),
                pl.col("chemotherapy"),
                pl.col("radio_therapy"),
                pl.col("hormone_therapy")
            ],
            separator="_",
        ).alias("her2_er_chemo_radio_hormone"),
    )

    df_C = np.array(df_C_all.select(context))

    if centered:
        X_mu = np.mean(df_X, axis=0)
        X_centered = df_X - X_mu
        return X_centered, df_Y, df_C


    return df_X, df_Y, df_C



class BRCADataset(Dataset):
    def __init__(self):
        self.X, self.Y, self.C = load_brca_data()

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.Y[idx])


class LinearRegression:
    def __init__(self):
        pass
    
    def fit(self, X, Y):
        X_bias = np.hstack((np.ones((X.shape[0], 1), dtype=X.dtype), X))
        self.coeffs = np.linalg.inv(X_bias.T @ X_bias) @ X_bias.T @ Y
        return self
    
    def predict(self, X):
        X_bias = np.hstack((np.ones((X.shape[0], 1), dtype=X.dtype), X))
        return X_bias @ self.coeffs
    

class LogisticRegression(torch.nn.Module):
    def __init__(self, n_features, output_dim):
        super().__init__()
        self.linear = torch.nn.Linear(n_features, output_dim)
    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs


class ContextLinearRegression:
    def __init__(self):
        self.models = dict()
    
    def fit(self, X, Y, C):
        self.context_labels = np.unique(C)
        for context in self.context_labels:
            X_i = X[(C == context).flatten()]
            Y_i = Y[(C == context).flatten()]
            model = LinearRegression().fit(X_i, Y_i)
            self.models[context] = model
        return self
    
    def predict(self, X, C):
        y_pred = np.zeros((len(X), 1))
        for label in self.context_labels:
            l_idx = (C == label).flatten()
            X_l = X[l_idx]
            y_pred[l_idx] = self.models[label].predict(X_l)
        return y_pred


class NeuralNetwork(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, n_hidden):
        super().__init__()
        self.flatten = nn.Flatten()
        
        self.first_layer = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nn.ReLU()
        )
        self.linear_relu_stack = nn.Sequential(*[
            nn.Sequential(
                    nn.Linear(dim_hidden, dim_hidden),
                    nn.ReLU()
                ) for _ in range(n_hidden - 1)
        ])
        self.last_layer = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = self.flatten(x)
        x = self.first_layer(x)
        x = self.linear_relu_stack(x)
        output = self.last_layer(x)
        return output


class LearnedContextNN(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, n_hidden, dim_context, n_context):
        super().__init__()

        self.dim_in = dim_in
        
        self.context = nn.Parameter(torch.rand(n_context, dim_context))
        self.flatten = nn.Flatten()
        
        self.first_layer = nn.Sequential(
            nn.Linear(dim_in + dim_context, dim_hidden),
            nn.ReLU()
        )
        self.linear_relu_stack = nn.Sequential(*[
            nn.Sequential(
                    nn.Linear(dim_hidden, dim_hidden),
                    nn.ReLU()
                ) for _ in range(n_hidden - 1)
        ])
        self.last_layer = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = self.flatten(x)
        device = x.device
        I = torch.eye(self.dim_in, device=device)
        block = torch.block_diag(I, self.context.to(device))
        x = torch.matmul(x, block)
        x = self.first_layer(x)
        x = self.linear_relu_stack(x)
        output = self.last_layer(x)
        return output


class LearnedContextSkipNN(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, n_hidden, dim_context, n_context):
        super().__init__()

        self.dim_in = dim_in
        
        self.context = nn.Parameter(torch.rand(n_context, dim_context))
        self.flatten = nn.Flatten()
        
        self.first_layer = nn.Sequential(
            nn.Linear(dim_in + dim_context, dim_hidden),
            nn.ReLU()
        )
        self.linear_relu_stack = nn.Sequential(*[
            nn.Sequential(
                    nn.Linear(dim_hidden, dim_hidden),
                    nn.ReLU()
                ) for _ in range(n_hidden - 1)
        ])
        self.last_layer = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = self.flatten(x)
        device = x.device
        I = torch.eye(self.dim_in, device=device)
        block = torch.block_diag(I, self.context.to(device))
        x = torch.matmul(x, block)
        x1 = self.first_layer(x)
        x2 = self.linear_relu_stack(x1)
        output = self.last_layer(x1 + x2)
        return output



def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn, device):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y)
    test_loss /= num_batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")
    return test_loss