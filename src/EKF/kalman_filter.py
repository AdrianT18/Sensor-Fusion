import numpy as np

"""
This script is the implementation of the Extended Kalman Filter.
In short the EKF is a non-linear version of the Kalman Filter. Which is used to estimate the state of a system. 
By updating the state with the measurement and the prediction. 
In this case I have used the Bicycle Model to track and predict the state.
"""


class ExtendedKalmanFilter:
    # The initial state is the state of the system at the start.
    def __init__(self, initial_state, state_covariance, process_noise, measurement_noise, wheelbase):
        self.state = initial_state
        self.covariance = state_covariance
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.wheelbase = wheelbase

    # Here we predict the next state based on control inputs and the time step.
    def predict(self, control_input, dt):
        # Extract steering angle and velocity from control input
        steering_angle, velocity = control_input

        # Current state variables
        x, y, psi, v = self.state

        # Predict new state using the motion model
        x_new = x + velocity * np.cos(psi) * dt
        y_new = y + velocity * np.sin(psi) * dt
        psi_new = psi + (velocity / self.wheelbase) * np.tan(steering_angle) * dt
        v_new = velocity

        # Update state
        self.state = np.array([x_new, y_new, psi_new, v_new])

        # Update covariance matrix which represents the uncertainty of the state
        F = self.jacobian_of_process_model(steering_angle, velocity, psi, dt)
        self.covariance = F @ self.covariance @ F.T + self.process_noise

    # Here we update the state estimator with a new measurement.
    def update(self, measurement):

        # Calculate Kalman gain
        H = self.jacobian_of_measurement_model()
        S = H @ self.covariance @ H.T + self.measurement_noise
        K = self.covariance @ H.T @ np.linalg.inv(S)

        # Update state
        y = measurement - self.measurement_model()
        self.state += K @ y

        # Update covariance matrix
        I = np.eye(self.covariance.shape[0])
        self.covariance = (I - K @ H) @ self.covariance

    # Here we process the model using the Bicycle Model. - TODO if you have control input
    def process_model(self, state, control_input, dt):
        pass

    # Return the position and velocity of the system.
    def measurement_model(self):
        return self.state[:2]

    # Here we calculate the Jacobian of the process model.
    def jacobian_of_process_model(self, steering_angle, velocity, psi, dt):
        # Compute partial derivatives for the Jacobian matrix
        a11 = 1
        a12 = 0
        a13 = -velocity * np.sin(psi) * dt
        a14 = np.cos(psi) * dt
        a21 = 0
        a22 = 1
        a23 = velocity * np.cos(psi) * dt
        a24 = np.sin(psi) * dt
        a31 = 0
        a32 = 0
        a33 = 1
        a34 = (1 / self.wheelbase) * np.tan(steering_angle) * dt
        a41 = 0
        a42 = 0
        a43 = 0
        a44 = 1

        J = np.array([[a11, a12, a13, a14],
                      [a21, a22, a23, a24],
                      [a31, a32, a33, a34],
                      [a41, a42, a43, a44]])
        return J

    # Computes the Jacobian matrix of the measurement model.
    def jacobian_of_measurement_model(self):
        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]])
        return H
