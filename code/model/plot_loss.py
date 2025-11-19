import pandas as pd
import matplotlib.pyplot as plt

loss_gan = pd.read_parquet('loss_history_gan_1.parquet')
loss_unc = pd.read_parquet('loss_history_unc_1.parquet')

# print(loss_gan.columns)
# print(loss_unc.columns)

# Unconditional train loss
plt.plot(loss_unc["epoch"], loss_unc["train_unc_loss"])
plt.xlabel("Epoch (Unc)")
plt.ylabel("Train Unc Loss")
plt.title("Unconditional Training Loss")
plt.show()

# GAN conditional & residual loss
plt.plot(loss_gan["epoch"], loss_gan["train_cond_loss"], label="cond_train")
plt.plot(loss_gan["epoch"], loss_gan["train_res_loss"], label="res_train")
plt.xlabel("Epoch (GAN)")
plt.ylabel("Loss")
plt.legend()
plt.title("GAN Phase Losses")
plt.show()

loss_gan = loss_gan.value_counts()
loss_unc = loss_unc.value_counts()

print(loss_gan)
print(loss_unc)
