"""Script to run in nengo to visualize the dynamics of a rotating body"""
import nengo
import numpy as np
from gyro_network import GyroNet, GyroEnv

def wrap_angle(t, theta):
    """Wrap the angular measurement to -pi:pi"""
    return theta - 2*np.pi * int((theta + np.pi)/2/np.pi)

model = nengo.Network()
with model:
    gnet = GyroNet()
    torque = nengo.Node([0])
    nengo.Connection(torque, gnet.input, synapse=None)

    env = nengo.Node(
        GyroEnv(
            size=10,
        ),
        size_in=1,
    )

    nengo.Connection(gnet.output[0], env, synapse=None)
    nengo.Connection(gnet.output[0], nengo.Node(wrap_angle, size_in=1, label="angular position"), synapse=None)
    nengo.Connection(gnet.output[1], nengo.Node(size_in=1, label="angular velocity"), synapse=None)
