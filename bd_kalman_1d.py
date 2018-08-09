import numpy as np
import matplotlib.pyplot as plt
import nengo
import nengo_brainstorm
from kalman import KalmanNet, LDSNet

SYS_TAU = 0.01
SIM_TIME = 10*SYS_TAU

DELTA_T = 0.001 # discretization time
A = np.array([[-1/SYS_TAU]])
B = np.array([[1/SYS_TAU]])
C = np.array([[1]])
D = np.zeros_like(A)
Q = np.array([[0.1]])
R = np.array([[0.01]])

fig, ax = plt.subplots(figsize=(8, 6))
# run underlying dynamical system beforehand
STIM_PARAMS = {0: 1, 5*SYS_TAU: 0}
def run_underlying_system()
    state_model = nengo.Network()
    with state_model:
        lds_net = LDSNet(A, B, C, D, Q, R)
        stim = nengo.Node(nengo.utils.functions.piecewise(STIM_PARAMS))
        nengo.Connection(stim, lds_net.input, synapse=None)
        stim_probe = nengo.Probe(stim)
        lds_state_probe = nengo.Probe(lds_net.state)
        lds_out_probe = nengo.Probe(lds_net.output)
    
    sim = nengo.Simulator(state_model)
    sim.run(SIM_TIME)
    # set up run with braindrop,
    trange = sim.trange().reshape(-1, 1)
    stim_data = dict(np.hstack((trange, sim.data[stim_probe])))
    measure_data = dict(np.hstack((trange, sim.data[lds_out_probe])))
    return stim_data, measure_data
STIM_DATA, MEASURE_DATA = run_underlying_system(A, B, C, D, Q, R, STIM_PARAMS)

# run with reference nengo
ref_model = nengo.Network()
with ref_model:
    knet = KalmanNet(256, A, B, C, Q, R, DELTA_T)
    stim_input = nengo.Node(nengo.utils.functions.piecewise(stim_data))
    stim_measure = nengo.Node(nengo.utils.functions.piecewise(measure_data))
    nengo.Connection(stim_input, knet.input_system, synapse=None)
    nengo.Connection(stim_measure, knet.input_system, synapse=None)
    probe_readout = nengo.Probe(knet.readout)
sim = nengo.Simulator(ref_model)
sim.run(SIM_TIME)
sim.data[probe_readout]

# run with braindrop
model = nengo.Network()
with model:
    knet = KalmanNet(256, A, B, C, Q, R, DELTA_T)
    stim_input = nengo.Node(nengo.utils.functions.piecewise(stim_data))
    stim_measure = nengo.Node(nengo.utils.functions.piecewise(measure_data))
    nengo.Connection(stim_input, knet.input_system, synapse=None)
    nengo.Connection(stim_measure, knet.input_system, synapse=None)
    probe_readout = nengo.Probe(knet.readout)

sim = nengo_brainstorm.Simulator(model)
sim.run(SIM_TIME)

