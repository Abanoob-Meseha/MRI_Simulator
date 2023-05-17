import matplotlib.pyplot as plt
import numpy as np


Rf_line = 20
Gz_line = 15
Gy_line = 10
Gx_line = 5
Ro_line = 0

[plt.axhline(y = i, color = 'r', linestyle = '-') for i in [Ro_line,Gx_line,Gy_line,Gz_line,Rf_line]]

x_rf1 = np.linspace(0, 20, 1000)
y_rf1 = Rf_line + ((3) * np.sinc(x_rf1 - 10))


x_rf2 = np.linspace(30, 50, 1000)
y_rf2 = Rf_line + ((3) * np.sinc(x_rf2 - 40))


plt.plot(x_rf1, y_rf1, color='maroon', marker='o')
plt.plot(x_rf2, y_rf2, color='maroon', marker='o')

plt.xlabel('t (msec)')
plt.ylabel('value')
plt.title("T2 preparation pulse")
plt.show()