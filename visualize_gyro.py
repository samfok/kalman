"""Script to run in nengo to visualize the dynamics of a rotating body"""
import nengo
import numpy as np
from gyro_network import GyroNet

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
