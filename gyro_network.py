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
    def __init__(self, inertia=1., g_tau=1., label="gyro"):
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

            accel = nengo.Node(lambda t, x: x, size_in=1, size_out=1)

            self.output= nengo.Node(lambda t, x: x, size_in=2, size_out=2, label="output")
            nengo.Connection(gyro_state, self.output, synapse=None)

class GyroEnv(object):
    def __init__(self, size=10, dt=0.001):

        self.size = size
        self.x = size / 2.
        self.y = size / 2.
        self.radius = size / 4. # distance from center of accelerometer

        # describe the shapes in the environment
        self.svg_open = (
            '<svg width="100%%" height="100%%" ' +
            'viewbox="0 0 {0} {1}">'.format(self.size, self.size))
        self.agent_template = (
            '<polygon points="0.25,0.25 -0.25,0.25 0,-0.5" style="fill:blue" ' +
            'transform="translate({0},{1}) rotate({2})"/>')
        self.accel_template = (
            '<circle cx="{0}" cy="{1}" ' + 'r="{}" '.format(self.radius/8) + 
            'fill="black"/>')
        self.svg_close = '</svg>'

        self._nengo_html_ = ''

    def __call__(self, t, x):
        theta = x[0]
        agent_direction = theta * 180. / np.pi + 90.

        self._nengo_html_ = self.svg_open
        self._nengo_html_ += self.agent_template.format(self.x, self.y, agent_direction)
        self._nengo_html_ += self.accel_template.format(self.x+np.cos(theta)*self.radius, self.y+np.sin(theta)*self.radius)
        self._nengo_html_ += self.svg_close
