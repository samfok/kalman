import numpy as np
import matplotlib.pyplot as plt
import nengo

from gyro_network import GyroNet

SIM_TIME = 5

def stim_fun(t, period=1., offset=0.5):
    """Stimulus function"""
    return 1 - 2*(int((t+offset) / period) % 2)

net = nengo.Network()
with net:
    gnet = GyroNet()

    torque = nengo.Node(lambda t: stim_fun(t), size_in=0, size_out=1)
    nengo.Connection(torque, gnet.input)

    t_probe = nengo.Probe(torque)
    g_probe = nengo.Probe(gnet.output)

sim = nengo.Simulator(net)
sim.run(SIM_TIME)
t_data = sim.data[t_probe]
g_data = sim.data[g_probe]
fig, axs = plt.subplots(nrows=3)
axs[0].plot(sim.trange(), t_data)
axs[1].plot(sim.trange(), g_data[:, 1])
axs[2].plot(sim.trange(), g_data[:, 0])

plt.show()
