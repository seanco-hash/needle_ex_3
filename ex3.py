import numpy as np

betha = 1
#
# v1 = np.array([[0.7,0.2,0.1],[0.25, 0.7, 0.1],[0.05, 0.2, 0.7]])
# v2 = np.array([[1/3],[1/3],[1/3]])
#
#
# print(np.dot(betha * v1, (1 - betha) * v2))
#
#
#
# while True:
#     pass

# At = np.full((6,6), (1 / 6))

# M = np.array([[1, 0, 1/4, 0, 0, 0],[0, 2/3, 1/4, 0, 0, 0],[0, 1/3, 1/4, 0, 0, 0], [0, 0, 0, 1/3, 1/3, 1/3],[0, 0, 1/4, 1/3, 1/3, 1/3],[0, 0, 0, 1/3, 1/3, 1/3]])
M = np.array([[0, 0, 0, 0],
              [1/3, 1, 0, 0],
              [1/3, 0, 1, 0],
              [1/3, 0, 0, 1]])

print(M)


# r = np.array([[1/6],[1/6],[1/6],[1/6],[1/6],[1/6]])
r = np.array([[1/4],[1/4],[1/4],[1/4]])

# Ak = np.zeros((6,6))
# Ak = np.zeros((4,4))
# Ak[0,:] = 1 / 3
# Ak[1,:] = 1 / 3
# Ak[2,:] = 1 / 3
Ak = np.full((4,4), 1/4)
M = betha * M + (1 - betha) * Ak

while True:
    r_n = np.dot(M, r)
    if np.linalg.norm(r - r_n) <= 0.01:
        break
    r = r_n

print(r)