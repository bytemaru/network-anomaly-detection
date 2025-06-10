import numpy as np

lg = [0.9425, 0.9406, 0.9422, 0.9414, 0.9422, 0.9443, 0.9415, 0.9421, 0.9420, 0.9450]
dt = [0.9453,  0.9472,  0.9467,  0.9455,  0.9456, 0.9475, 0.9461, 0.9523, 0.9531, 0.9433]

lg_variance = np.var(lg)
dt_variance = np.var(dt)

print(lg_variance)
print(dt_variance)