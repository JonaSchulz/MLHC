import numpy as np
import matplotlib.pyplot as plt


data = np.load("loss_files/loss_224_long.npz")
train_loss = data["train_loss"]
val_loss = data["val_loss"]
print(len(train_loss))
plt.plot(train_loss)
plt.show()
