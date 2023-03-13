# CT phantom
from phantominator import shepp_logan
import matplotlib.pyplot as plt

phantomImg = shepp_logan(128)
# MR phantom (returns proton density, T1, and T2 maps)
PD, T1, T2 = shepp_logan((128, 128, 20), MR=True)
fig, ax = plt.subplots()
ax.imshow(M0[:,:,15], cmap='gray')
plt.show()