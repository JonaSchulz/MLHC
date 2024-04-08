import numpy as np
import matplotlib.pyplot as plt


data = np.load("losses_new.npz")
train_loss = data["train_loss"]
val_loss = data["val_loss"]
test_loss = data["test_loss"]

plt.plot(train_loss)
plt.show()
