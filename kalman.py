"""Defines Kalman filter and dynamical system implementations
"""
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
        P_predict = np.linalg.multi_dot([A, P, A.T]) + Q
        K_process = np.dot(P_predict, C.T)
        K_measure = np.linalg.multi_dot([C, P_predict, C.T]) + R
        K = np.dot(K_process, np.linalg.inv(K_measure))
        P = np.dot(I_NN-np.dot(K, C), P_predict)
        diff = np.sum(np.abs(K - K_prev)) / entries
        iter_count += 1
        K_prev = K

    if dbg:
        print(iter_count, diff)
    return K

def pass_fun(t, x):
    return x

def c_to_d_kf(A_CT, B_CT, Q_CT, R_CT, dt):
    """Convert continuous form LDS equations into their discrete form"""
    A_DT = dt * A_CT + np.eye(A_CT.shape[0])
    B_DT = dt * B_CT

    # Q_DT = Q_CT*dt
    # R_DT = R_CT

    # Q_DT = Q_CT*dt
    # R_DT = R_CT/dt

    Q_DT = Q_CT
    R_DT = R_CT/dt

    return A_DT, B_DT, Q_DT, R_DT

class KalmanNetDT(nengo.Network):
    """A Kalman filter nengo Network built for the discrete time system

    x[t] = A_DTx[t-1] + B_DTu[t] + v[t]
    y[t] = C_DTx[t] + w[t]

    v[t] ~ normal(0, Q_DT)
    w[t] ~ normal(0, R_DT)

    Parameters
    ----------
    neurons : int
        number of neurons
    A_DT: NxN numpy array
        System dynamics
        Describes how the previous state mixes to generate the current state
    B_DT: NxL numpy array
        System input matrix
        Describes how the inputs mix to drive the system
    C_DT: MxN numpy array
        Measurement matrix
        Describes how the system's dimensions mix to produce the output measurement
    Q_DT: NxN numpy array
        Intrinsic noise covariance matrix
    R_DT: MxM numpy array
        Measurement noise covariance matrix
    tau_syn: float (optional)
        Synaptic time constant
    P0: NxN numpy array (optional)
        Initial error covariance
    dt: float (optional)
        time step used to discretize system
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
    readout: nengo Node
        the state readout
    """
    def __init__(self, neurons, A_DT, B_DT, C_DT, Q_DT, R_DT,
                 tau_syn=0.01, dt=0.001,
                 neuron_type=nengo.neurons.LIF(), label="KalmanNetwork", verbose=False):
        super(KalmanNetDT, self).__init__(label=label)
        M, N = C_DT.shape
        L = B_DT.shape[1]

        P0 = np.zeros_like(A_DT)
        # Kalman Filter steady-state form
        # xhat[t] = A_K xhat[t-1] + B_K u[t-1] + K_ss y[t]
        if np.all(Q_DT == 0) and np.all(R_DT == 0): # handle no-noise case
            K_ss = np.eye(N, M)
        else:
            K_ss = find_k_ss(A_DT, C_DT, Q_DT, R_DT, P0)
        A_K = np.dot(np.eye(N) - np.dot(K_ss, C_DT), A_DT)
        B_K = np.dot(np.eye(N) - np.dot(K_ss, C_DT), B_DT)

        # Convert to continuous time form
        # x[t] = xdot dt + x[t-1]
        # dx/dt = A_CT xhat[t-1] + B_CT u[t-1] + K_ss_CT y[t]
        A_CT = (A_K - np.eye(N)) / dt
        B_CT = B_K / dt
        K_ss_CT = K_ss / dt

        # Convert to NEF matrices
        A_NEF = tau_syn * A_CT + np.eye(N)
        B_NEF = tau_syn * B_CT
        K_NEF = tau_syn * K_ss_CT

        if verbose:
            print("A_DT\n", A_DT)
            print("B_DT\n", B_DT)
            print("C_DT\n", C_DT)
            print("A_K\n", A_K)
            print("B_K\n", A_K)
            print("K_SS\n", K_ss)
            print("A_CT\n", A_CT)
            print("B_CT\n", A_CT)
            print("K_SS_CT\n", K_ss_CT)
            print("A_NEF\n", A_NEF)
            print("B_NEF\n", B_NEF)
            print("K_NEF\n", K_NEF)

        with self:
            self.input_system = nengo.Node(pass_fun, size_in=L)
            self.input_measurement = nengo.Node(pass_fun, size_in=M)
            self.state = nengo.Ensemble(neurons, N, neuron_type=neuron_type)
            self.readout = nengo.Node(pass_fun, size_in=N)

            nengo.Connection(
                self.input_system, self.state, function=lambda x:np.dot(B_NEF, x), synapse=tau_syn)
            nengo.Connection(
                self.input_measurement, self.state, function=lambda x:np.dot(K_NEF, x),
                synapse=tau_syn)
            nengo.Connection(
                self.state, self.state, function=lambda x: np.dot(A_NEF, x), synapse=tau_syn)

            # nengo.Connection(self.input_system, self.state, transform=B_NEF, synapse=tau_syn)
            # nengo.Connection(self.input_measurement, self.state, transform=K_NEF, synapse=tau_syn)
            # nengo.Connection(self.state, self.state, transform=A_NEF, synapse=tau_syn)

            nengo.Connection(
                self.input_system, self.readout,
                function=lambda x: np.dot(B_NEF, x), synapse=tau_syn)
            nengo.Connection(
                self.input_measurement, self.readout,
                function=lambda x:np.dot(K_NEF, x), synapse=tau_syn)
            nengo.Connection(
                self.state, self.readout,
                function=lambda x: np.dot(A_NEF, x), synapse=tau_syn)

            # nengo.Connection(self.input_system, self.readout, transform=B_NEF, synapse=tau_syn)
            # nengo.Connection(
            #     self.input_measurement, self.readout, transform=K_NEF, synapse=tau_syn)
            # nengo.Connection(self.state, self.readout, transform=A_NEF, synapse=tau_syn)

class KalmanNet(KalmanNetDT):
    """A Kalman filter nengo Network built for the continuous time system

    dx/dt = Ax + Bu + v
    y = Cx + w

    v(t) ~ normal(0, Q)
    w(t) ~ normal(0, R)

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
        time step used to discretize system
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
    readout: nengo Node
        the state readout
    """
    def __init__(self, neurons, A, B, C, Q, R, dt,
                 tau_syn=0.01,
                 neuron_type=nengo.neurons.LIF(), label="KalmanNetwork", verbose=False):
        A_DT, B_DT, Q_DT, R_DT = c_to_d_kf(A, B, Q, R, dt)
        C_DT = C
        if verbose:
            print("A", A)
            print("B", B)
            print("C", C)
            print("Q", Q)
            print("R", R)
        super(KalmanNet, self).__init__(
            neurons, A_DT, B_DT, C_DT, Q_DT, R_DT,
            dt=dt, tau_syn=tau_syn, neuron_type=neuron_type, label=label, verbose=verbose)

def make_random_fun(mean, cov, dt):
    """Generate a function that creates random noise"""
    inv_sqrtdt = 1./np.sqrt(dt)
    def add_random_noise(t):
        """Adds random noise to a vector"""
        return inv_sqrtdt * np.random.multivariate_normal(mean, cov)
    return add_random_noise

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
    def __init__(self, A, B, C, D=None, Q=None, R=None, tau_syn=1.0, dt=0.001, label="LDSNet"):
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
            if Q is not None: # add noise if present
                self.process_noise = nengo.Node(make_random_fun(np.zeros(N), Q, dt))
                nengo.Connection(
                    self.process_noise, self.state, transform=tau_syn, synapse=tau_syn)

            # connect readout
            nengo.Connection(self.state, self.output, transform=C, synapse=None)
            if D is not None:
                nengo.Connection(self.input, self.output, transform=D, synapse=None)
            if R is not None:
                self.output_noise = nengo.Node(make_random_fun(np.zeros(M), R, dt=1))
                nengo.Connection(self.output_noise, self.output, synapse=None)
