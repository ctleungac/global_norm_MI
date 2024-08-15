import numpy as np

class KSGEstimator:
    def __init__(self, mode='cpu', kraskov_k=3):
        self.mode = mode
        self.kraskov_k = kraskov_k
        self.ksg_est = self._initialize_estimator()

    def _initialize_estimator(self):
        if self.mode == 'gpu':
            import idtxl.estimators_opencl as est
            settings = {'kraskov_k': self.kraskov_k}
            return est.OpenCLKraskovMI(settings = settings)
        elif self.mode == 'cpu':
            from . import knnie
            return knnie.kraskov_mi
        else:
            raise ValueError("Mode should be either 'cpu' or 'gpu'.")

    # @staticmethod
    # def global_norm(data, C):
    #     data = np.reshape(data, (data.shape[0], int(data.size / data.shape[0])))
    #     means = np.mean(data, axis=0)
    #     data = data - means
    #     norm = np.sqrt(np.mean(np.sum(data ** 2, axis=1)))
    #     data *= C / norm
    #     return data, norm ** 2

    @staticmethod
    def local_norm(data, C):
        data = np.reshape(data, (data.shape[0], int(data.size / data.shape[0])))
        means = np.mean(data, axis=0)
        data = data - means
        norm = np.tile(np.sqrt(np.mean(data ** 2, axis=0)), (data.shape[0], 1))
        normalized_data = C * data / (norm + 1e-7)
        return normalized_data, norm ** 2

    @staticmethod
    def global_normalize_inf_norm(data_orig, C):
        data = np.reshape(data_orig, (data_orig.shape[0], int(data_orig.size / data_orig.shape[0])))
        means = np.mean(data, axis=0)
        data = np.abs(data - means)
        norm = np.mean(np.max(data, axis=1))
        data = data_orig * C / norm
        return data

    def global_norm_ksg_estimator(self, X, Y, C_y_min=0.5, C_y_max=2, step=0.1):
        C_range = np.arange(C_y_min, C_y_max, step)
        normx = self.global_normalize_inf_norm(X, C=1)
        MI_final = 0
        for C_y in C_range:
            normy = self.global_normalize_inf_norm(Y, C=C_y)
            if self.mode == 'gpu':
                MI = self.ksg_est.estimate(normx, normy)
            else:
                MI = self.ksg_est(normx, normy,k=self.kraskov_k)
            MI = np.array(MI).flatten()[0]
            MI_final = max(MI_final, MI)
        return MI_final
    
    def local_norm_ksg_estimator(self, X, Y):
        normx = self.local_norm(X, C=1)[0]
        normy = self.local_norm(Y, C=1)[0]
        if self.mode == 'gpu':
            MI = self.ksg_est.estimate(normx, normy)
        else:
            MI = self.ksg_est(normx, normy,k=self.kraskov_k)
        MI = np.array(MI).flatten()[0]
        return MI


# Example usage
if __name__ == "__main__":
    X = np.random.rand(100, 10)
    Y = np.random.rand(100, 10)

    ksg = KSGEstimator(mode='gpu', kraskov_k=3)
    mi_result = ksg.global_norm_ksg_estimator(X, Y)
    print(f"Estimated Mutual Information: {mi_result}")