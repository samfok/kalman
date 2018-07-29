import numpy as np
import nengo

class Kalman:
    """Kalman filter

    The Kalman filter estimates the state of an N dimensional linear dynamical system
    from measurements over time
    using an internal model of the system dynamics
    and knowledge (or assumptions, more likely) about
    the intrinsic system noise and measurement noise

    System follows
    x[t] = Ax[t-1] + Bu[t-1] + w
    z[t] = Hx[t] + v

    x is the N-dimensional state vector
    u is the L-dimensional input vector
    w is the L-dimensional process noise vector
      u~Normal(0, Q)
    z is the M-dimensional measurement vector
    v is the M-dimensional measurement noise vector
      v~Normal(0, R)

    Parameters
    ----------
    A: NxN numpy array
        System dynamics
        Describes how the previous state mixes to generate the current state
    B: NxL numpy array
        System input matrix
        Describes how the inputs mix to drive the system
    C: MxN numpy array
        Measurement matrix
        Describes how the system's dimensions mix to produce the output measurement
    Q: NxN numpy array
        Intrinsic noise covariance matrix
    R: MxM numpy array
        Measurement noise covariance matrix
    """
    def __init__(self, A, B, C, Q, R, xhat0, phat0):
        self.A = A
        self.B = B
        self.C = C
        self.Q = Q
        self.R = R

        N = A.shape[0]
        L = B.shape[1]
        M = C.shape[0]

        assert A.shape == (N, N)
        assert B.shape == (N, L)
        assert Q.shape == (N, N)
        assert R.shape == (M, M)

        self.xhat_predict = np.zeros(N)
        self.xhat = xhat0
        self.p_predict = np.zeros((N, N))
        self.p = phat0

        self.z_predict = np.zeros(L)
        self.K_process = np.zeros((N, M))
        self.K_measure = np.zeros((M, N))
        self.K = np.zeros((N, M))
        self.eye = np.eye(N)

    def step(self, u, z):
        """Step the Kalman filter forward in time"""
        self._predict(u)
        self._update(z)
        return self.xhat, self.p, self.K

    def _predict(self, u):
        """Predict the current state from the previous state estimate"""
        self.xhat_predict = np.dot(self.A, self.xhat) + np.dot(self.B, u)
        self.p_predict = np.dot(self.A, np.dot(self.p, self.A.T)) + self.Q
        self.z_predict = np.dot(self.C, self.xhat_predict)

    def _update(self, z):
        """Update the state estimate from the current prediction and current measurement"""
        self.K_process = np.dot(self.p_predict, self.C.T)
        self.K_measure = np.dot(self.C, np.dot(self.p_predict, self.C.T)) + self.R
        self.K = np.dot(self.K_process, np.linalg.inv(self.K_measure))
        self.xhat = self.xhat_predict + np.dot(self.K, z-self.z_predict)
        self.p = np.dot(self.eye - np.dot(self.K, self.C), self.p_predict)

class LDS:
    """Linear dynamical system

    With optional stochastic inputs

    Follows
      x[t] = Ax[t-1] + Bu[t-1] + w[t-1]
      y[t] = Cx[t] + Du[t] + v[t]

    Parameters
    ----------
    A: NxN numpy array
        System dynamics
        Describes how the previous state mixes to generate the current state
    B: NxL numpy array
        System input matrix
        Describes how the inputs mix to drive the system
    C: MxN numpy array
        Measurement matrix
        Describes how the system's dimensions mix to produce the output measurement
    D: MxL numpy array
        Bypass matrix
        Describes how the inputs mix to contribute to the output measurement
    Q: NxN numpy array
        Intrinsic noise covariance matrix
    R: MxM numpy array
        Measurement noise covariance matrix
    """
    def __init__(self, A, B, C, D, Q, R, x0):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.Q = Q
        self.R = R

        N = A.shape[0]
        L = B.shape[1]
        M = C.shape[0]

        assert A.shape == (N, N), ("Expected shape {}, got shape {}".format((N, N), A.shape))
        assert B.shape == (N, L), ("Expected shape {}, got shape {}".format((N, L), B.shape))
        assert C.shape == (M, N), ("Expected shape {}, got shape {}".format((M, N), C.shape))
        assert D.shape == (M, L), ("Expected shape {}, got shape {}".format((M, L), D.shape))
        assert Q.shape == (N, N), ("Expected shape {}, got shape {}".format((N, N), Q.shape))
        assert R.shape == (M, M), ("Expected shape {}, got shape {}".format((M, M), R.shape))

        assert x0.shape == (N,)

        self.w_mu = np.zeros(N)
        self.v_mu = np.zeros(M)

        self.x0 = x0.copy()
        self.y0 = np.dot(self.C, self.x0) + np.random.multivariate_normal(self.v_mu, self.R)
        self.x = x0.copy()

    def step(self, u):
        """Step the dynamical system forward in time"""
        w = np.random.multivariate_normal(self.w_mu, self.Q)
        v = np.random.multivariate_normal(self.v_mu, self.R)

        self.x = np.dot(self.A, self.x) + np.dot(self.B, u) + w
        y = np.dot(self.C, self.x) + np.dot(self.D, u) + v
        return self.x, y

    @property
    def initial_condition(self):
        """Return the initial condition of the LDS"""
        return self.x0, self.y0

    @property
    def state_dimensions(self):
        """Get the dimensionality of the system"""
        return self.A.shape[0]

    @property
    def output_dimensions(self):
        """Get the dimensionality of the system"""
        return self.C.shape[0]

def find_k_ss(A, C, Q, R, P0, tol=1E-5, max_iter=1000, dbg=False):
    """Iteratively finds the steady state Kalman gain

    Parameters
    ----------
    A: NxN numpy array
        System dynamics
        Describes how the previous state mixes to generate the current state
    C: MxN numpy array
        Measurement matrix
        Describes how the system's dimensions mix to produce the output measurement
    Q: NxN numpy array
        Intrinsic noise covariance matrix
    R: MxM numpy array
        Measurement noise covariance matrix
    P0: NxN numpy array
        Initial error covariance
    """
    P = P0
    M, N = C.shape

    I_NN = np.eye(N) # NxN identity matrix
    K_prev = np.ones((N, M))
    entries = N*M

    iter_count = 0
    diff = 2*tol
    while iter_count > max_iter or diff > tol:
        P_predict = np.dot(A, np.dot(P, A.T)) + Q
        K_process = np.dot(P_predict, C.T)
        K_measure = np.dot(C, np.dot(P_predict, C.T)) + R
        K = np.dot(K_process, np.linalg.inv(K_measure))
        P = np.dot(I_NN-np.dot(K, C), P_predict)
        diff = np.sum(np.abs(K - K_prev)) / entries
        iter_count += 1
        K_prev = K

    if dbg:
        print(iter_count, diff)
    return K

def solve_k_ss(A, C, Q, R):
    """Finds the steady state Kalman gain by analytically"""
    # I'm not sure this is the correct formula, do we have a citation or derivation?
    # I think it's something like the first iteration of the Kalman gain given P0=0 -SF
    R_inv = np.linalg.inv(R)
    K_den = np.eye(A.shape[0])+np.dot(Q, np.dot(C.T, np.dot(R_inv, C)))
    K_num = np.dot(Q, np.dot(C.T, R_inv))
    K = np.dot(np.linalg.inv(K_den), K_num)
    return K

def pass_fun(t, x):
    return x

def add_random_noise(t, mean, cov):
    """Adds random noise to a vector"""
    noise = np.random.multivariate_normal(mean, cov)
    return noise

def c_to_d_kf(A_CT, B_CT, Q_CT, dt=0.001):
    """Convert continuous form LDS equations into their discrete form"""
    A_DT = dt * A_CT + np.eye(A_CT.shape[0])
    B_DT = dt * B_CT
    Q_DT = Q_CT * dt
    return A_DT, B_DT, Q_DT

class KalmanNet(nengo.Network):
    """A Kalman filter nengo Network

    Parameters
    ----------
    neurons : int
        number of neurons
    A: NxN numpy array
        System dynamics
        Describes how the previous state mixes to generate the current state
    B: NxL numpy array
        System input matrix
        Describes how the inputs mix to drive the system
    C: MxN numpy array
        Measurement matrix
        Describes how the system's dimensions mix to produce the output measurement
    Q: NxN numpy array
        Intrinsic noise covariance matrix
    R: MxM numpy array
        Measurement noise covariance matrix
    tau_syn: float (optional)
        Synaptic time constant
    P0: NxN numpy array (optional)
        Initial error covariance
    dt: float (optional)
        time step used ot discretize system
    neuron_type: nengo neuron model instance (optional)
        e.g. nengo.neurons.Direct() for doing "just the math"
    label: string (optional)
        label for network

    Attributes
    ----------
    input_measurement : nengo Node
        measurement y
    state: nengo Node
        the state estimate
    input_system : nengo Node (if B provided)
        input u of the system
    """
    def __init__(self, neurons, A, B, C, Q, R,
                 tau_syn=0.01, P0=0, dt=0.001,
                 neuron_type=nengo.neurons.LIF(), label="KalmanNetwork"):
        super(KalmanNet, self).__init__(label=label)
        M, N = C.shape
        L = B.shape[1]
        # print("System A", A)
        # print("System B", B)
        # print("System C", C)

        K_ss = find_k_ss(A, C, Q, R, P0)

        # Kalman Filter steady-state form
        # xhat[t] = A xhat[t-1] + B u[t-1] + K_ss y[t]
        A = np.dot(np.eye(N) - np.dot(K_ss, C), A)
        B = np.dot(np.eye(N) - np.dot(K_ss, C), B)
        # print("A", A)
        # print("B", A)
        # print("K_SS", K_ss)

        # Convert to continuous time form
        # x[t] = xdot dt + x[t-1]
        # dx/dt = A_CT xhat[t-1] + B_CT u[t-1] + K_ss_CT y[t]
        A_CT = (A - np.eye(N)) / dt
        B_CT = B / dt
        K_ss_CT = K_ss / dt
        # print("A_CT", A_CT)
        # print("B_CT", A_CT)
        # print("K_SS_CT", K_ss_CT)

        # Convert to NEF matrices
        A_NEF = tau_syn * A_CT + np.eye(N)
        B_NEF = tau_syn * B_CT
        K_NEF = tau_syn * K_ss_CT
        # print("A_NEF", A_NEF)
        # print("B_NEF", A_NEF)
        # print("K_NEF", K_NEF)

        with self:
            self.input_system = nengo.Node(pass_fun, size_in=L)
            self.input_measurement = nengo.Node(pass_fun, size_in=M)
            self.state = nengo.Ensemble(neurons, N, neuron_type=neuron_type)

            nengo.Connection(self.input_system, self.state, transform=B_NEF, synapse=tau_syn)
            nengo.Connection(self.input_measurement, self.state, transform=K_NEF, synapse=tau_syn)
            nengo.Connection(self.state, self.state, transform=A_NEF, synapse=tau_syn)

            self.readout = nengo.Node(pass_fun, size_in=N)
            nengo.Connection(self.input_system, self.readout, transform=B_NEF, synapse=tau_syn)
            nengo.Connection(self.input_measurement, self.readout, transform=K_NEF, synapse=tau_syn)
            nengo.Connection(self.state, self.readout, transform=A_NEF, synapse=tau_syn)

class LDSNet(nengo.Network):
    """Implements an linear dynamical system with noise

    With optional stochastic inputs

    Follows
      dx/dt = Ax + Bu + w
      y = Cx + Du + v

    Parameters
    ----------
    A: NxN numpy array
        System dynamics
        Describes how the previous state mixes to generate the current state
    B: NxL numpy array
        System input matrix
        Describes how the inputs mix to drive the system
    C: MxN numpy array
        Measurement matrix
        Describes how the system's dimensions mix to produce the output measurement
    D: MxL numpy array (optional)
        Bypass matrix
        Describes how the inputs mix to contribute to the output measurement
    Q: NxN numpy array (optional)
        Intrinsic noise covariance matrix
    R: MxM numpy array (optional)
        Measurement noise covariance matrix

    Attributes
    ----------
    input:  nengo Node
        provides the input u
    state: nengo Node
        maintains the state x
    output: nengo Node
        provides the output y
    """
    def __init__(self, A, B, C, D=None, Q=None, R=None, tau_syn=0.1, label="LDSNet"):
        super(LDSNet, self).__init__(label=label)
        N = A.shape[0]
        L = B.shape[1]
        M = C.shape[0]

        B_NEF = tau_syn * B
        A_NEF = tau_syn * A + np.eye(N)
        with self:
            self.input = nengo.Node(pass_fun, size_in=L)
            self.state = nengo.Node(pass_fun, size_in=N)
            self.output = nengo.Node(pass_fun, size_in=M)

            # connect core dynamics
            nengo.Connection(self.input, self.state, transform=B_NEF, synapse=tau_syn)
            nengo.Connection(self.state, self.state, transform=A_NEF, synapse=tau_syn)
            if Q: # add noise if present
                self.process_noise = nengo.Node(lambda t: add_random_noise(t, np.zeros(N), Q))
                nengo.Connection(
                    self.process_noise, self.state, transform=tau_syn, synapse=tau_syn)

            # connect readout
            nengo.Connection(self.state, self.output, transform=C, synapse=None)
            if D:
                nengo.Connection(self.input, self.output, transform=D, synapse=None)
            if R:
                self.output_noise = nengo.Node(lambda t: add_random_noise(t, np.zeros(M), R))
                nengo.Connection(self.output_noise, self.output, synapse=None)
