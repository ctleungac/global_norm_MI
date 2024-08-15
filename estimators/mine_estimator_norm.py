import numpy as np


class MINEEstimator:
    def __init__(self, num_epoch=50):
        self.num_epoch = num_epoch
        self.mine_est = self._initialize_estimator()

    def _initialize_estimator(self):
        from . import MINE_estimate as mine
        return mine

    @staticmethod
    def local_norm(data, C):
        data = np.reshape(data, (data.shape[0], int(data.size / data.shape[0])))
        means = np.mean(data, axis=0)
        data = data - means
        norm = np.tile(np.sqrt(np.mean(data ** 2, axis=0)), (data.shape[0], 1))
        normalized_data = C * data / (norm + 1e-7)
        return normalized_data, norm ** 2

    @staticmethod
    def global_normalize_mine(data_orig, C, p=2, dim_correction=False):
        data = np.reshape(data_orig, (data_orig.shape[0], int(data_orig.size / data_orig.shape[0])))
        means = np.mean(data, axis=0)
        data = np.abs(data - means)
        norm = np.sqrt(np.mean(np.sum(data ** 2, axis=1)))
        if dim_correction:
            norm /= np.sqrt(data.shape[1])
        data = C * (data_orig - means) / norm
        return data

    def global_norm_mine_estimator(self, X, Y):
        normx = self.global_normalize_mine(X, C=1)
        normy = self.global_normalize_mine(Y, C=1)
        MI = self.mine_est.MINE_MI(normx, normy, total_epochs=self.num_epoch)
        return float(MI.flatten()[0])

    def local_norm_mine_estimator(self, X, Y):
        normx = self.local_norm(X, C=1)[0]
        normy = self.local_norm(Y, C=1)[0]
        MI = self.mine_est.MINE_MI(normx, normy, total_epochs=self.num_epoch)
        return float(MI.flatten()[0])


# Example usage
if __name__ == "__main__":
    X = np.random.rand(100, 10)
    Y = np.random.rand(100, 10)

    mine_estimator = MINEEstimator(num_epoch=50)
    mi_result_global = mine_estimator.global_norm_mine_estimator(X, Y)
    mi_result_local = mine_estimator.local_norm_mine_estimator(X, Y)

    print(f"Estimated Mutual Information (Global Norm): {mi_result_global}")
    print(f"Estimated Mutual Information (Local Norm): {mi_result_local}")