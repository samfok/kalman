import nengo_brainstorm
from kalman import KalmanNet, LDSNet

SYS_TAU = 0.01
SIM_TIME = 10*SYS_TAU

DELTA_T = 0.001 # discretization time
A = np.array([[-1/SYS_TAU]])
B = np.array([[1/SYS_TAU]])
C = np.array([[1]])
Q = np.array([[0.1]])
R = np.array([[0.01]])

ref_model = nengo.network()
with ref_model:
    lds_net = LDSNet

model = nengo.Network()
with model:
    knet = KalmanNet(256, A, B, C, Q, R
