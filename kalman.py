import numpy as np

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
    def __init__(self, A, B, H, Q, R, xhat0, Phat0):
        self.A = A
        self.B = B
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
        self.xhat = np.zeros(N)
        self.p_predict = np.zeros((N, N))
        self.p = np.zeros((N, N))

        self.z_predict = np.zeros(L)
        self.K_process = np.zeros((N, M))
        self.K_measure = np.zeros((M, N))
        self.K = np.zeros((N, M))
        self.eye = np.eye(N)

    def step(self, z):
        """Step the Kalman filter forward in time"""
        self._predict()
        self._update(z)
        return self.xhat, self.p, self.K

    def _predict(self, z):
        """Predict the current state from the previous state estimate"""
        self.xhat_predict = np.dot(self.A, self.xhat) + np.dot(self.B, u)
        self.p_predict = np.dot(self.A, np.dot(self.p, self.A.T)) + self.Q
        self.z_predict = np.dot(self.H, self.xhat_predict)

    def _update(self, z):
        """Update the state estimate from the current prediction and current measurement"""
        self.K_process = np.dot(self.p_predict, self.H.T)
        self.K_measure = np.dot(self.H, np.dot(self.p_predict, self.H.T)) + self.R
        self.K = np.dot(self.K_process, np.linalg.inv(self.K_measure))
        self.xhat = xhat_predict + np.dot(self.K, z-self.z_predict)
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
    def __init__(self, A, B, C, D, x0, Q, R):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.Q = Q
        self.R = R

        N = A.shape[0]
        L = B.shape[1]
        M = C.shape[0]

        assert A.shape == (N, N) 
        assert B.shape == (N, L)
        assert C.shape == (M, N)
        assert D.shape == (M, L)
        assert Q.shape == (N, N)
        assert R.shape == (M, M)

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
        return A.shape[0]

    @property
    def output_dimensions(self):
        """Get the dimensionality of the system"""
        return C.shape[0]
