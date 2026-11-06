import numpy as np

def compute_participation_ratio(cov_matrix):
    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    participation_ratio = (np.sum(eigenvalues)**2) / np.sum(eigenvalues**2)
    return participation_ratio

def estimate_grassmann_distance(W_1, W_2, print_results=False):

    Q_1, _ = np.linalg.qr(W_1, mode="reduced")
    Q_2, _ = np.linalg.qr(W_2, mode="reduced")

    # Compute principal angles
    M = Q_1.T @ Q_2
    _, s, _ = np.linalg.svd(M)
    s = np.clip(s, -1.0, 1.0)  # numerical stability
    principal_angles = np.arccos(s)

    grassmann_distance = (2/np.pi) * np.sqrt(np.mean(principal_angles**2))

    if print_results:
        print("angles (deg):", principal_angles * 180 / np.pi)
        print("normalized d:", grassmann_distance)

    return grassmann_distance

def estimate_projection_distance(W_true, W_est, print_results=False):

    nb_neurons, nb_latent = W_true.shape

    # Orthonormalize the columns of W_true and W_est using QR decomposition
    Q_true, _ = np.linalg.qr(W_true, mode="reduced")
    Q_est, _ = np.linalg.qr(W_est, mode="reduced")

    # Project the true subspace onto the estimated subspace
    P_est = Q_est @ Q_est.T
    I = np.eye(nb_neurons)
    normalized_distance = (1/nb_latent) * np.linalg.norm((I - P_est) @ Q_true, ord='fro')**2

    return normalized_distance