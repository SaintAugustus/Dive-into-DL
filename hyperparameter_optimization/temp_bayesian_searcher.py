from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import minimize

from hyperparameter_optimization.hyperopt_api import HPOSearcher


class BayesianOptimizationSearcher(HPOSearcher):
    def __init__(self, config_space: dict, init_config=None, n_initial_points=5):
        self.config_space = config_space
        self.init_config = init_config if init_config is not None else self._init_random_config(n_initial_points)
        self.gp = GaussianProcessRegressor(kernel=ConstantKernel(1.0)*RBF(length_scale=1.0))
        self.X = []
        self.Y = []
        self.save_hyperparameters()

    def _init_random_config(self, n):
        # Initialize n random configurations
        return [
            {
                name: domain.rvs()
                for name, domain in self.config_space.items()
            }
            for _ in range(n)
        ]

    def sample_configuration(self):
        if self.init_config:
            result = self.init_config.pop(0)
        else:
            # Find the configuration that maximizes the acquisition function
            result = minimize(lambda x: -self.acquisition_function(x), x0=None, bounds=self.config_space).x
        return result

    def acquisition_function(self, config):
        # Implement an acquisition function here. The Expected Improvement (EI) function is commonly used.
        pass

    def update(self, config: dict, error: float, additional_info=None):
        # Update the Gaussian process with the result of evaluating the function with the given config
        self.X.append(list(config.values()))
        self.Y.append(error)
        self.gp.fit(self.X, self.Y)
