# test_bilstm.py

import numpy as np
from models.bilstm_model import build_bilstm_model, train_bilstm_model

# Fake data for testing
X_train = np.random.rand(100, 60, 1)  # 100 samples, 60 timesteps, 1 feature
y_train = np.random.rand(100)

# Build model
model = build_bilstm_model((60,1))

# Train for 2 epochs just to test
train_bilstm_model(model, X_train, y_train, X_val=None, y_val=None)