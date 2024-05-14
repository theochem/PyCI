import numpy as np 
from abc import ABCMeta, abstractmethod
class Projection(metaclass=ABCMeta):
    
    @abstractmethod
    def __init__(self, initial_guess,constraints):
        self.initial_guess = initial_guess
        self.constraints = constraints
    
    @abstractmethod 
    def optimize(self):
        pass

class Dykstra(metaclass=ABCMeta):
    def __init__(self, initial_guess:float, constraints:list, alpha:int =1, max_iterations:int =100, eps:int =1e-6):
        """
        Dykstra's algorithm for projection onto convex sets.

        Parameters
        ----------
        initial_guess : float
            Initializing with guessed density matrix \Gamma_0.
        constraints : list
            A list of projection operators onto all $J$ of the constraints as a single set, $\{\mathcal{P}_j\}_{j=0}^{J-1}$.
        alpha : int, optional
            Tunning parameter, by default 1
        max_iterations : int, optional
            Number of maximum iterations, by default 100
        eps : int, optional
            Tolerance, by default 1e-6
        """        
        super().__init__(initial_guess, constraints)
        self.alpha = alpha
        self.max_iterations = max_iterations
        self.eps = eps
    def optimize(self):
        X = [np.zeros(self.initial_guess) for i in range(len(self.constraints))]
        D = np.copy(self.initial_guess)
        norm = []
        for i in range (self.max_iterations):
            for j, projection in enumerate(self.constraints):
                C = D - X[j]
                D = projection(C)
                X[j] = D - C
            norm.append(np.linalg.norm(D - self.initial_guess))
            is_stop = self.alpha * abs(norm[i] - norm[i - 1]) + (1 - self.alpha) * norm[i] < self.eps
            if is_stop:
                break
class Neumann(metaclass=ABCMeta):
    def __init__(self, initial_guess, constraints, alpha, max_iterations=100, eps=1e-6):
        super().__init__(initial_guess, constraints)
        self.alpha = alpha
        self.max_iterations = max_iterations
        self.eps = eps
    def optimize(self):
        D = np.copy(self.initial_guess)
        norm = []
        for i in range (self.max_iterations):
            for projection in enumerate(self.constraints):
                D = projection(D)
            norm.append(np.linalg.norm(D - self.initial_guess))
            is_stop = self.alpha * abs(norm[i] - norm[i - 1]) + (1 - self.alpha) * norm[i] < self.eps
            if is_stop:
                break

