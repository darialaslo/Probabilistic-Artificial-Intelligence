import random
import os
import typing
import logging
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import matplotlib.pyplot as plt

#extra imports
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.stats import norm

EXTENDED_EVALUATION = False
# Set `EXTENDED_EVALUATION` to `True` in order to visualize your predictions.

#setting random seed
np.random.seed(0)


""" Solution """


class BO_algo(object):
    def __init__(self):
        """Initializes the algorithm with a parameter configuration. """

        # TODO: enter your code here
        self.previous_points = []
        # IMPORTANT: DO NOT REMOVE THOSE ATTRIBUTES AND USE sklearn.gaussian_process.GaussianProcessRegressor instances!
        # Otherwise, the extended evaluation will break.
        ######
        #defining kernels
        self.constraint_kernel = ConstantKernel(3.5, constant_value_bounds="fixed") * RBF(2.0,length_scale_bounds="fixed")
        self.objective_kernel = ConstantKernel(1.5, constant_value_bounds="fixed") * RBF(1.5,length_scale_bounds="fixed")
        #defining the noise params
        self.constraint_noise = 0.005
        self.objective_noise = 0.01
        ######
        # self.constraint_model = None  # TODO : GP model for the constraint function
        # self.objective_model = None  # TODO : GP model for your acquisition function
        self.constraint_model = GaussianProcessRegressor(kernel = self.constraint_kernel, alpha = self.constraint_noise**2)  # TODO : GP model for the constraint function
        self.objective_model = GaussianProcessRegressor(kernel = self.objective_kernel, alpha = self.objective_noise**2) # TODO : GP model for your acquisition function

    def next_recommendation(self) -> np.ndarray:
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: np.ndarray
            1 x domain.shape[0] array containing the next point to evaluate
        """

        # TODO: enter your code here
        # In implementing this function, you may use optimize_acquisition_function() defined below.
        #####
        #if there is no point sampled yet, we cannot optimize the acquisition function so we have to sample an initial point
        if len(self.previous_points)==0 :
            #sample random point
            #considering the shape of the domain
            return np.array([[np.random.uniform(0,6), np.random.uniform(0,6)]])

        #if there are points optimize the acquisition function
        else:
            return self.optimize_acquisition_function()




    def optimize_acquisition_function(self) -> np.ndarray:  # DON'T MODIFY THIS FUNCTION
        """
        Optimizes the acquisition function.

        Returns
        -------
        x_opt: np.ndarray
            1 x domain.shape[0] array containing the point that approximately maximizes the acquisition function.
        """

        def objective(x: np.array):
            return - self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            x0 = domain_x[0, 0] + (domain_x[0, 1] - domain_x[0, 0]) * \
                 np.random.rand(1)
            x1 = domain_x[1, 0] + (domain_x[1, 1] - domain_x[1, 0]) * \
                 np.random.rand(1)
            result = fmin_l_bfgs_b(objective, x0=np.array([x0, x1]), bounds=domain_x,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *domain_x[0]))
            f_values.append(result[1])

        ind = np.argmin(f_values)
        return np.atleast_2d(x_values[ind])

    def acquisition_function(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the acquisition function.

        Parameters
        ----------
        x: np.ndarray
            point in the domain of f

        Returns
        ------
        af_value: float
            value of the acquisition function at x
        """

        # TODO: enter your code here
    

        expectimprov = self.expected_improvement(x)
        constraint_weight = self.constraint_function(x)

        return float(expectimprov * constraint_weight)

        raise NotImplementedError

    def expected_improvement(self, x):
        """
        Compute expected improvement at point x based on the Gaussian
        Process surrogate. 

        Parameters
        ---------------
            x: Point at which expected improvement should be computed

        Returns
        ---------------
        the expected improvement 
        """
       
        mu, sigma = self.objective_model.predict(x.reshape(1, -1), return_std=True)

        #if there are no samples yet, just sample from the Gaussian Process
        if len(self.previous_points)==0:
            opt = self.objective_model.sample_y(x.reshape(1, -1))
        else:
            #get the point for which the minimum is obtained 
            ind = np.argmin(np.array([i[2] for i in self.previous_points]))
            opt = np.array(self.previous_points)[ind, 2] 
        #0.1 trade off variable 
        Z = (opt - mu - 0.1)/ sigma

        return (opt - mu - 0.1) * norm.cdf(Z) + norm.pdf(Z)*sigma

    def constraint_function(self, x):
        """
        Based on : https://arxiv.org/abs/1403.5607

        Parameters
        ---------------
            x: Points at which the constraint is evaluated

        Returns
        ---------------
        the probability that the value of the condition using the 
        surrogate will be lower than 0 
        """

        mu, sigma = self.constraint_model.predict(x.reshape(1,-1), return_std=True)
        weight = norm.cdf(0, loc=mu, scale=sigma)

        return weight

    def add_data_point(self, x: np.ndarray, z: float, c: float):
        """
        Add data points to the model.

        Parameters
        ----------
        x: np.ndarray
            point in the domain of f
        z: np.ndarray
            value of the acquisition function at x
        c: np.ndarray
            value of the condition function at x
        """

        assert x.shape == (1, 2)
        self.previous_points.append([float(x[:, 0]), float(x[:, 1]), float(z), float(c)])
        # TODO: enter your code here
        #check if this is the first point added or not
        #if yes, fit the model
        if len(self.previous_points) == 1:
                self.constraint_model.fit(x,c)
                self.objective_model.fit(x,z)
        else: 
            #retrain and update
            self.constraint_model.fit(X = np.array(self.previous_points)[:,:2], y = np.array(self.previous_points)[:,3])
            self.objective_model.fit(X = np.array(self.previous_points)[:,:2], y = np.array(self.previous_points)[:,2])


        #raise NotImplementedError

    def get_solution(self) -> np.ndarray:
        """
        Return x_opt that is believed to be the minimizer of f.

        Returns
        -------
        solution: np.ndarray
            1 x domain.shape[0] array containing the optimal solution of the problem
        """

        # TODO: enter your code here
        #extract the function values
        z = np.array(self.previous_points)[:,2]
        #check for which ones the constraint is satistified and if it is not set the function to a very high value
        z[np.array(self.previous_points)[:,3] > 0] = 1e6

        #get the index at which the lowest value of the function is obtained
        if len(self.previous_points)==0:
            ind = np.argmin(z)

        else:
            ind = np.argmin(z)

        #extract the coordinates of the point which obtainest the lowest value for the function
        xstar = np.array(self.previous_points)[ind, :2]

        return xstar
        raise NotImplementedError


""" 
    Toy problem to check  you code works as expected
    IMPORTANT: This example is never used and has nothing in common with the task you
    are evaluated on, it's here only for development and illustration purposes.
"""
domain_x = np.array([[0, 6], [0, 6]])
EVALUATION_GRID_POINTS = 250
CONSTRAINT_OFFSET = - 0.8  # This is an offset you can change to make the constraint more or less difficult to fulfill
LAMBDA = 0.0  # You shouldn't change this value


def check_in_domain(x) -> bool:
    """Validate input"""
    x = np.atleast_2d(x)
    v_dim_0 = np.all(x[:, 0] >= domain_x[0, 0]) and np.all(x[:, 0] <= domain_x[0, 1])
    v_dim_1 = np.all(x[:, 1] >= domain_x[1, 0]) and np.all(x[:, 0] <= domain_x[1, 1])

    return v_dim_0 and v_dim_1


def f(x) -> np.ndarray:
    """Dummy objective"""
    l1 = lambda x0, x1: np.sin(x0) + x1 - 1

    return l1(x[:, 0], x[:, 1])


def c(x) -> np.ndarray:
    """Dummy constraint"""
    c1 = lambda x, y: np.cos(x) * np.cos(y) - 0.1

    return c1(x[:, 0], x[:, 1]) - CONSTRAINT_OFFSET


def get_valid_opt(f, c, domain) -> typing.Tuple[float, float, np.ndarray, np.ndarray]:
    nx, ny = (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS)
    x = np.linspace(domain[0, 0], domain[0, 1], nx)
    y = np.linspace(domain[1, 0], domain[1, 1], ny)
    xv, yv = np.meshgrid(x, y)
    z = np.array([xv.reshape(-1), yv.reshape(-1)]).T

    true_values = f(z)
    true_cond = c(z)
    valid_data_idx = np.where(true_cond < LAMBDA)[0]
    f_opt = np.min(true_values[np.where(true_cond < LAMBDA)])
    x_opt = z[valid_data_idx][np.argmin(true_values[np.where(true_cond < LAMBDA)])]
    f_max = np.max(np.abs(true_values))
    x_max = np.argmax(np.abs(true_values))
    return f_opt, f_max, x_opt, x_max


def perform_extended_evaluation(agent, output_dir='./'):
    fig = plt.figure(figsize=(25, 5), dpi=50)
    nx, ny = (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS)
    x = np.linspace(0.0, 6.0, nx)
    y = np.linspace(0.0, 6.0, ny)
    xv, yv = np.meshgrid(x, y)
    x_b, y_b = agent.get_solution()
    z = np.array([xv.reshape(-1), yv.reshape(-1)]).T
    predictions, stds = agent.objective_model.predict(z, return_std=True)
    predictions = np.reshape(predictions, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    conds = agent.constraint_model.predict(z)
    conds = np.reshape(conds, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))

    true_values = f(z)
    true_cond = c(z)
    conditions_verif = (true_cond < LAMBDA).astype(float)
    conditions_with_nans = 1 - np.copy(conditions_verif)
    conditions_with_nans[np.where(conditions_with_nans == 0)] = np.nan
    conditions_with_nans = np.reshape(conditions_with_nans, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    valid_data_idx = np.where(true_cond < LAMBDA)[0]

    f_opt = np.min(true_values[np.where(true_cond < LAMBDA)])
    x_opt = z[valid_data_idx][np.argmin(true_values[np.where(true_cond < LAMBDA)])]

    sampled_point = np.array(agent.previous_points)

    ax_condition = fig.add_subplot(1, 4, 4)
    im_cond = ax_condition.pcolormesh(xv, yv, conds.reshape((EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS)), shading='auto',
                                      linewidth=0)
    im_cond.set_edgecolor('face')
    fig.colorbar(im_cond, ax=ax_condition)
    ax_condition.scatter(sampled_point[:, 0], sampled_point[:, 1], cmap='Blues', marker='x',
                         label='Sampled Point by BO', antialiased=True, linewidth=0)
    ax_condition.pcolormesh(xv, yv, conditions_with_nans, shading='auto', cmap='Reds', alpha=0.7, vmin=0, vmax=1.0,
                            linewidth=0, antialiased=True)
    ax_condition.set_title('Constraint GP Posterior +  True Constraint (Red is Infeasible)')
    ax_condition.legend(fontsize='x-small')

    ax_gp_f = fig.add_subplot(1, 4, 2, projection='3d')
    ax_gp_f.plot_surface(
        X=xv,
        Y=yv,
        Z=predictions,
        rcount=100,
        ccount=100,
        linewidth=0,
        antialiased=False
    )
    ax_gp_f.set_title('Posterior 3D for Objective')

    ax_gp_c = fig.add_subplot(1, 4, 3, projection='3d')
    ax_gp_c.plot_surface(
        X=xv,
        Y=yv,
        Z=conds,
        rcount=100,
        ccount=100,
        linewidth=0,
        antialiased=False
    )
    ax_gp_c.set_title('Posterior 3D for Constraint')

    ax_predictions = fig.add_subplot(1, 4, 1)
    im_predictions = ax_predictions.pcolormesh(xv, yv, predictions, shading='auto', label='Posterior',linewidth=0, antialiased=True)
    im_predictions.set_edgecolor('face')
    fig.colorbar(im_predictions, ax=ax_predictions)
    ax_predictions.pcolormesh(xv, yv, conditions_with_nans, shading='auto', cmap='Reds', alpha=0.7, vmin=0, vmax=1.0,
                              label=' True Infeasible',linewidth=0, antialiased=True)
    ax_predictions.scatter(x_b, y_b, s=20, marker='x', label='Predicted Value by BO')
    ax_predictions.scatter(x_opt[0], x_opt[1], s=20, marker='o', label='True Optimimum Under Constraint')
    ax_predictions.set_title('Objective GP Posterior + True Constraint (Red is Infeasible)')
    ax_predictions.legend(fontsize='x-small')
    figure_path = os.path.join(output_dir, 'extended_evaluation.pdf')
    fig.savefig(figure_path)
    plt.show()


def train_on_toy(agent, iteration):
    logging.info('Running model on toy example.')
    seed = 1234
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    for j in range(iteration):
        # Get next recommendation
        x = agent.next_recommendation()

        # Check for valid shape
        assert x.shape == (1, domain_x.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {domain_x.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x) + np.random.normal(size=(x.shape[0],), scale=0.01)
        cost_val = c(x) + np.random.normal(size=(x.shape[0],), scale=0.005)
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = np.atleast_2d(agent.get_solution())
    assert solution.shape == (1, domain_x.shape[0]), \
        f"The function get solution must return a numpy array of shape (" \
        f"1, {domain_x.shape[0]})"

    assert check_in_domain(solution), \
        f'The function get solution must return a point within the ' \
        f'domain, {solution} returned instead'

    # Compute regret
    f_opt, f_max, x_opt, x_max = get_valid_opt(f, c, domain_x)
    if c(solution) > 0.0:
        regret = 1
    else:
        regret = (f(solution) - f_opt) / f_max

    print(f'Optimal value: {f_opt}\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret{regret}')
    return agent



def main():
    logging.warning(
        'This main method is for illustrative purposes only and will NEVER be called by the checker!\n'
        'The checker always calls run_solution directly.\n'
        'Please implement your solution exclusively in the methods and classes mentioned in the task description.'
    )

    seed = 1234
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    agent = BO_algo()

    agent = train_on_toy(agent, 20)

    if EXTENDED_EVALUATION:
        perform_extended_evaluation(agent)


if __name__ == "__main__":
    main()
