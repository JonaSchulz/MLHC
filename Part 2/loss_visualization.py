import numpy as np
import matplotlib.pyplot as plt


data = np.load("losses_224_2.npz")
train_loss = data["train_loss"]
val_loss = data["val_loss"]
test_loss = data["test_loss"]

loss = np.convolve(test_loss, np.ones(50) / 50)

plt.plot(loss)
plt.show()
