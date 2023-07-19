import os
import sys
import numpy as np
import random
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


# class for transformer network
class Transformer(nn.Module):
    # define and intialize the structure of the neural network
    def __init__(
        self,
        input_dim=7,
        model_dim=1000,
        output_dim=1000,
        n_heads=4,
        dim_feedforward=1000,
        n_layers=4,
        learning_rate=0.00005,
        n_head_layers=2,
        head_norm=False,
        dropout=0.1,
        opt="adam",
    ):
        super().__init__()
        # define hyperparameters
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.dim_feedforward = dim_feedforward
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.n_head_layers = n_head_layers
        self.head_norm = head_norm
        self.dropout = dropout
        # define subnetworks
        self.embedding = nn.Linear(input_dim, model_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                model_dim, n_heads, dim_feedforward=dim_feedforward, dropout=dropout
            ),
            n_layers,
        )
        # head_layers have output_dim
        if n_head_layers == 0:
            self.head_layers = []
        else:
            if head_norm:
                self.norm_layers = nn.ModuleList([nn.LayerNorm(model_dim)])
            self.head_layers = nn.ModuleList([nn.Linear(model_dim, output_dim)])
            for i in range(n_head_layers - 1):
                if head_norm:
                    self.norm_layers.append(nn.LayerNorm(output_dim))
                self.head_layers.append(nn.Linear(output_dim, output_dim))
        # option to use adam or sgd
        if opt == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        if opt == "sgdca" or opt == "sgdslr" or opt == "sgd":
            self.optimizer = torch.optim.SGD(
                self.parameters(), lr=self.learning_rate, momentum=0.9
            )

    def forward(self, view1, view2, mask=None, mult_reps=False):
        """
        the two views are shaped like DataBatch(x=[12329, 7], y=[256], batch=[12329], ptr=[257])
        transformer expects (sequence_length, feature_number) so we don't need to transpose
        """
        view1_features, view2_features = [], []
        assert len(view1) == len(view2)
        for i in range(len(view1)):
            jet1 = view1[i]
            jet2 = view2[i]
            # make a copy
            x_1 = jet1.x + 0.0
            x_2 = jet2.x + 0.0
            # cast to torch.float32 to prevent RuntimeError: mat1 and mat2 must have the same dtype
            x_1 = x_1.to(torch.float32)
            x_2 = x_2.to(torch.float32)
            # embedding
            x_1 = self.embedding(x_1)
            x_2 = self.embedding(x_2)
            # transformer
            x_1 = self.transformer(x_1, mask=mask)
            x_2 = self.transformer(x_2, mask=mask)
            # sum over sequence dim
            # (batch_size, model_dim)
            x_1 = x_1.sum(0)
            x_2 = x_2.sum(0)
            # head
            x_1 = self.head(x_1, mult_reps)
            x_2 = self.head(x_2, mult_reps)
            # append to feature list
            view1_features.append(x_1)
            view2_features.append(x_2)

        return torch.stack(view1_features, view2_features)

    def head(self, x, mult_reps):
        """
        calculates output of the head if it exists, i.e. if n_head_layer>0
        returns multiple representation layers if asked for by mult_reps = True
        input:  x shape=(batchsize, model_dim)
                mult_reps boolean
        output: reps shape=(batchsize, output_dim)                  for mult_reps=False
                reps shape=(batchsize, number_of_reps, output_dim)  for mult_reps=True
        """
        relu = nn.ReLU()
        # return representations from multiple layers for evaluation
        if mult_reps == True:
            if self.n_head_layers > 0:
                reps = torch.empty(x.shape[0], self.n_head_layers + 1, self.output_dim)
                reps[:, 0] = x
                for i, layer in enumerate(self.head_layers):
                    # only apply layer norm on head if chosen
                    if self.head_norm:
                        x = self.norm_layers[i](x)
                    x = relu(x)
                    x = layer(x)
                    reps[:, i + 1] = x
                # shape (n_head_layers, output_dim)
                return reps
            # no head exists -> just return x in a list with dimension 1
            else:
                reps = x[:, None, :]
                # shape (batchsize, 1, model_dim)
                return reps
        # return only last representation for contrastive loss
        else:
            for i, layer in enumerate(
                self.head_layers
            ):  # will do nothing if n_head_layers is 0
                if self.head_norm:
                    x = self.norm_layers[i](x)
                x = relu(x)
                x = layer(x)
            # shape either (model_dim) if no head, or (output_dim) if head exists
            return x
