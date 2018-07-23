import numpy as np
import nengo

class GyroNet(nengo.Network):
    """Create a network that mimics the dynamics of a gyro

    Parameters
    ----------
    inertia: float
        The inertia of the gyro
    g_tau: float
        The tau used internally within the gyro network

    Attributes
    ----------
    input : Node
        provide input to the gyro
    output: Node
        returns output from the gyro
        dim 0: angular position
        dim 1: angular velocity
    """
    def __init__(self, inertia, g_tau, label="gyro"):
        super(GyroNet, self).__init__(label)

        A = np.array([[0, 1], [0, 0]])
        B = np.array([0, 1./inertia])
        g_A = g_tau*A + np.eye(A.shape[0])
        g_B = (g_tau*B)[1]

        with self:
            self.input = nengo.Node(lambda t, x: x, size_in=1, size_out=1, label="input")

            gyro_state = nengo.Node(lambda t, x: x, size_in=2, size_out=2)
            nengo.Connection(self.input, gyro_state[1], transform=g_B, synapse=g_tau)
            nengo.Connection(gyro_state, gyro_state, transform=g_A, synapse=g_tau)

            # accel = nengo.Node(lambda t, x: x, size_in=1, size_out=1)

            self.output= nengo.Node(lambda t, x: x, size_in=2, size_out=2, label="output")
            nengo.Connection(gyro_state, self.output, synapse=None)

