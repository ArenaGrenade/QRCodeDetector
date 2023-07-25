import cv2
import numpy as np

def initialize_kalman_filter(state_dim, measurement_dim):
    kalman_filter = cv2.KalmanFilter(state_dim, measurement_dim)
    kalman_filter.transitionMatrix = np.eye(state_dim, dtype=np.float32)  # State transition matrix
    kalman_filter.measurementMatrix = np.eye(measurement_dim, state_dim, dtype=np.float32)  # Measurement matrix
    kalman_filter.processNoiseCov = 1e-3 * np.eye(state_dim, dtype=np.float32)  # Process noise covariance
    kalman_filter.measurementNoiseCov = 1e-2 * np.eye(measurement_dim, dtype=np.float32)  # Measurement noise covariance
    kalman_filter.errorCovPost = 1e-1 * np.eye(state_dim, dtype=np.float32)  # Initial covariance
    return kalman_filter
