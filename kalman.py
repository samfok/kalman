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
    H: MxN numpy array
        Measurement matrix
        Describes how the system's dimensions mix to produce the output measurement
    Q: NxN numpy array
        Intrinsic noise covariance matrix
    R: MxM numpy array
        Measurement noise covariance matrix
    """
    def __init__(self, A, B, H, Q, R, xhat0, phat0):
        self.A = A
        self.B = B
        self.H = H
        self.Q = Q
        self.R = R

        N = A.shape[0]
        L = B.shape[1]
        M = H.shape[0]

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
        self.z_predict = np.dot(self.H, self.xhat_predict)

    def _update(self, z):
        """Update the state estimate from the current prediction and current measurement"""
        self.K_process = np.dot(self.p_predict, self.H.T)
        self.K_measure = np.dot(self.H, np.dot(self.p_predict, self.H.T)) + self.R
        self.K = np.dot(self.K_process, np.linalg.inv(self.K_measure))
        self.xhat = self.xhat_predict + np.dot(self.K, z-self.z_predict)
        self.p = np.dot(self.eye - np.dot(self.K, self.H), self.p_predict)

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

def find_k_ss(A, C, Q, R, P0, tol=1E-5, max_iter=1000):
    """Iteratively finds the steady state Kalman gain

    Parameters
    ----------
    A : NxN numpy array
        System dynamics
        Describes how the previous state mixes to generate the current state
    C : MxN numpy array
        Measurement matrix
        Describes how the system's dimensions mix to produce the output measurement
    Q : NxN numpy array
        Intrinsic noise covariance matrix
    R : MxM numpy array
        Measurement noise covariance matrix
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
        P = np.dot(np.dot(I_NN-np.dot(K, C)), P_predict)
        diff = np.sum(np.abs(K - K_prev)) / entries
        iter_count += 1
        K_prev = K
    print(iter_count, diff)

    return K

def solve_k_ss(A, C, Q, R, P0):
    """Finds the steady state Kalman gain by analytically"""
    # I'm not sure this is the correct formula, do we have a citation or derivation? - SF
    R_inv = np.linalg.inv(R)
    K_den = np.eye+np.dot(Q, np.dot(C, np.dot(R_inv, C)))
    K_num = np.dot(Q, np.dot(C.T, R_inv))
    K = np.dot(np.linalg.inv(K_den), K_num)
    return K

class KalmanNetwork(nengo.Network):
    """A Kalman filter nengo Network

    Parameters
    ----------
    neurons : int
        number of neurons
    A : NxN numpy array
        System dynamics
        Describes how the previous state mixes to generate the current state
    B: NxL numpy array
        System input matrix
        Describes how the inputs mix to drive the system
    C : MxN numpy array
        Measurement matrix
        Describes how the system's dimensions mix to produce the output measurement
    Q : NxN numpy array
        Intrinsic noise covariance matrix
    R : MxM numpy array
        Measurement noise covariance matrix

    Attributes
    ----------
    input_y : nengo Node
        measurement y
    output : nengo Node
        delivers the output state estimate
    input_u : nengo Node
        if B provided
            input u of the system (0 by default)
    """
    def __init__(
            self, neurons, A, C, Q, R,
            tau_syn=0.01, P0=0, delta_t=0.001, B=None, label="KalmanNetwork"):
        super(KalmanNetwork, self).__init__(label=label)
        M, N = C.shape
        L = B.shape[1]

        self.K = find_k_ss(A, C, Q, R, P0)

        """
        Kalman filter update is described in discrete time

        with steady state kalman gain
            xhat[t] = (I-KC)xhat[t-1] + Ky[t]
        convert to continuous time
            in general,
            dx/dt = (x[t]- x[t-1]) / delta_t so x[t] = dx/dt delta_t + x[t-1]
            therefore
                dxhat/dt = -KC/delta_t xhat + K/delta_t y
        convert to NEF style feedback and feedforward gains
        dxhat/dt = -tau/delta_t KC xhat + tau/delta_t K y
        """
        A_NEF = -tau_syn/delta_t * np.dot(self.K, C)
        B_NEF = tau_syn/delta_t * self.K

        with self:
            self.input_y = nengo.Node(lambda t, x: x, size_in=M)
            self.ens = nengo.Ensemble(neurons, N)
            self.output = nengo.Node(lambda t, x: x, size_in=N)
            if B:
                B_input = tau_syn/delta_t * B
                self.input_u = nengo.Node(lambda t, x: x, size_in=L)
                nengo.Connection(self.input_u, self.ens, transform=B_input, synapse=tau_syn)
            nengo.Connection(self.input_y, self.ens, transform=B_NEF, synapse=tau_syn)
            nengo.Connection(self.ens, self.ens, transform=A_NEF, synapse=tau_syn)
