import math
T1 = 800
TR = 3000

Ernst_angle = round(math.degrees(math.acos(math.exp(-TR/T1))), 3)
print(Ernst_angle)