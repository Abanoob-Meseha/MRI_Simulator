import matplotlib.pyplot as plt
import numpy as np


Rf_line = 20
Gz_line = 15
Gy_line = 10
Gx_line = 5
Ro_line = 0

[plt.axhline(y = i, color = 'r', linestyle = '-') for i in [Ro_line,Gx_line,Gy_line,Gz_line,Rf_line]]

x_rf = np.linspace(0, 20, 1000)
y_rf = Rf_line + ((5) * np.sinc(x_rf - 10))

plt.step(x=[0, 20, 20], y=[Gz_line, (Gz_line + 1) * 1.06, Gz_line])

plt.plot(x_rf, y_rf, color='maroon', marker='o')

plt.xlabel('t (msec)')
plt.ylabel('value')
plt.title("180° rf inversion pulse")
plt.show()