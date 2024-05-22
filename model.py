import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import math

def dot_product_decode(Z):
	A_pred = torch.sigmoid(torch.matmul(Z,Z.t()))
	return A_pred

def glorot_init(input_dim, output_dim):
	init_range = np.sqrt(6.0/(input_dim + output_dim))
	initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
	return nn.Parameter(initial)

class GINConv(torch.nn.Module):
	def __init__(self, input_dim, output_dim):
		super().__init__()
		self.linear = torch.nn.Linear(input_dim, output_dim)

	def forward(self, A, X):
		X = self.linear(X + A @ X)
		X = torch.nn.functional.relu(X)
		return X


class GIN(torch.nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim, n_layers,
				 use_input_augmentation):
		super().__init__()
		self.in_proj = torch.nn.Linear(input_dim, hidden_dim)
		self.convs = torch.nn.ModuleList()
		self.use_input_agumentation = use_input_augmentation
		if(use_input_augmentation):
			self.hidden_input_dim = input_dim+hidden_dim
		else:
			self.hidden_input_dim = hidden_dim
		for _ in range(n_layers):
			self.convs.append(GINConv(self.hidden_input_dim, hidden_dim))
		self.out_proj = torch.nn.Linear(hidden_dim * (1 + n_layers), output_dim)

	def forward(self, A, X):
		initial_X = torch.empty_like(X).copy_(X)
		X = self.in_proj(X)
		hidden_states = [X]
		for layer in self.convs:
			if(self.use_input_agumentation):
				X = layer(A, torch.cat([initial_X,X],dim=1))
			else:
				X = layer(A, X)
			hidden_states.append(X)
		X = torch.cat(hidden_states, dim=1)
		X = self.out_proj(X)
		return X

class GAE(nn.Module):
	def __init__(self, num_hidden_layers, input_dim, hidden_dim, output_dim, activation,
				 use_input_augmentation, use_output_augmentation, encoder, variational=False):
		super(GAE,self).__init__()
		self.use_input_augmentation =- use_input_augmentation
		self.use_output_augmentation = use_output_augmentation
		self.encoder = encoder
		self.variational = variational
		self.base_gcn = GIN(input_dim, hidden_dim, output_dim, num_hidden_layers+2,
								use_input_augmentation = use_input_augmentation)

	def encode(self, initial_X, adj):
		hidden = self.base_gcn(adj, initial_X)
		return hidden

	def forward(self, initial_X, adj):
		Z = self.encode(initial_X, adj)
		return Z
