import numpy as np
import matplotlib.pyplot as plt


data = np.load("losses.npz")
train_loss = data["train_loss"]
val_loss = data["val_loss"]

plt.plot(val_loss)
plt.show()
