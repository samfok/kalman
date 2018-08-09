import numpy as np
import matplotlib.pyplot as plt
import nengo
import nengo_brainstorm
from kalman import KalmanNet, LDSNet

SYS_TAU = 0.01
SIM_TIME = 50*SYS_TAU

DELTA_T = 0.001 # discretization time
A = np.array([[-1/SYS_TAU]])
B = np.array([[1/SYS_TAU]])
C = np.array([[1]])
D = np.zeros_like(A)
Q = np.array([[0.1]])
R = np.array([[0.001]])

STIM_PARAMS = {
    0: 1,
    10*SYS_TAU: 0,
    20*SYS_TAU: 1,
    30*SYS_TAU: 0,
    40*SYS_TAU: 1,
}

fig, ax = plt.subplots(figsize=(16, 12))
# run underlying dynamical system beforehand
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
# plot underlying dynamical system results
ax.plot(sim.trange(), sim.data[stim_probe], alpha=0.5, label="stim")
ax.plot(sim.trange(), sim.data[lds_state_probe], alpha=0.5, label="system state")
ax.plot(sim.trange(), sim.data[lds_out_probe], alpha=0.3, label="measurement")

# set up run with braindrop,
trange = sim.trange().reshape(-1, 1)
measure_data = dict(np.hstack((trange, sim.data[lds_out_probe])))

# run with reference nengo
ref_model = nengo.Network()
with ref_model:
    stim_input = nengo.Node(nengo.utils.functions.piecewise(STIM_PARAMS))
    stim_measure = nengo.Node(nengo.utils.functions.piecewise(measure_data))

    knet_ref = KalmanNet(1, A, B, C, Q, R, DELTA_T, neuron_type=nengo.neurons.Direct())
    knet_spiking = KalmanNet(256, A, B, C, Q, R, DELTA_T)

    nengo.Connection(stim_input, knet_ref.input_system, synapse=None)
    nengo.Connection(stim_input, knet_spiking.input_system, synapse=None)
    nengo.Connection(stim_measure, knet_ref.input_measurement, synapse=None)
    nengo.Connection(stim_measure, knet_spiking.input_measurement, synapse=None)

    probe_readout_ref = nengo.Probe(knet_ref.readout)
    probe_readout_spiking = nengo.Probe(knet_spiking.readout)

sim = nengo.Simulator(ref_model)
sim.run(SIM_TIME)

# plot underlying dynamical system results
ax.plot(sim.trange(), sim.data[probe_readout_ref], alpha=0.5, label="KF reference")
ax.plot(sim.trange(), sim.data[probe_readout_spiking], alpha=0.5,
        label="KF {} simulated nengo neurons".format(knet_spiking.state.n_neurons))

# # run with braindrop
# model = nengo.Network()
# with model:
#     knet = KalmanNet(256, A, B, C, Q, R, DELTA_T)
#     stim_input = nengo.Node(nengo.utils.functions.piecewise(STIM_PARAMS))
#     stim_measure = nengo.Node(nengo.utils.functions.piecewise(measure_data))
#     nengo.Connection(stim_input, knet.input_system, synapse=None)
#     nengo.Connection(stim_measure, knet.input_system, synapse=None)
#     probe_readout = nengo.Probe(knet.readout)
# 
# sim = nengo_brainstorm.Simulator(model)
# sim.run(SIM_TIME)

ax.legend(loc="best")
plt.show()
