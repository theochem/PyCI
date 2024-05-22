import numpy as np 
from abc import ABC, abstractmethod

class Projection(ABC):
    
    def __init__(self, initial_guess: np.ndarray, constraints: list) -> None:
        r"""
        Initialize the Projection object.

        Parameters
        ----------
        initial guess: np.ndarray
            Initial with guess density matrix \Gamma_0.
        constraints: list
            List of projection operators $J$ onto convex subspaces C_i related to the constraints.
        """
        self.initial_guess = initial_guess
        self.constraints = constraints
    
    @abstractmethod 
    def optimize(self):
        r"""
        Algorithm to find the optimal wave function that satisfies all the constraints. This wave function Gamma is the
        one that minimizes ||Gamma-Gamma_0|| in the convex set C, which is the set of (approximately) N-representable density
        matrices      
        
        Parameters
        ----------
        Specific parameters depend on the particular algorithm
        """
        pass

class Dykstra(Projection):
    
    def __init__(self, initial_guess:np.ndarray, constraints:list, alpha:float =1.0, max_iterations:int =100, eps:float =1e-6) -> None:
        r"""
        Dykstra's class for projection onto convex sets.

        Parameters
        ----------
        initial_guess : np.ndarray
            Initial  guess for the density matrix \Gamma_0.
        constraints : list
            List of projection operators $J$ onto convex subspaces C_i related to the constraints.
        alpha : float
            Tunning parameter, by default 1
        max_iterations : int
            Number of maximum iterations, by default 100
        eps : float, optional
            Tolerance, by default 1e-6
        """        
        super().__init__(initial_guess, constraints)
        self.alpha = alpha
        self.max_iterations = max_iterations
        self.eps = eps
    
    def optimize(self) -> np.ndarray:
        r"""
        Dykstra's algorithm to find the optimal wave function that satisfies all the constraints        
        
        Returns
        ----------
        D: np.ndarray
            Optimal rdm that satisfies the given constraints
        """
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
        return D

class Neumann(Dykstra):
    
    def __init__(self, initial_guess:np.ndarray, constraints:list, alpha:float =1.0, max_iterations:int =100, eps:float =1e-6) -> None:
        r"""
        Neumann's class for projection onto convex sets.

        Parameters
        ----------
        initial_guess : np.ndarray
            Initial guess for the density matrix \Gamma_0.
        constraints : list
            List of projection operators $J$ onto convex subspaces C_i related to the constraints.
        alpha : float
            Tunning parameter, by default 1
        max_iterations : int, optional
            Number of maximum iterations, by default 100
        eps : float
            Tolerance, by default 1e-6
        """       
        super().__init__(initial_guess, constraints, alpha, max_iterations,eps) 
    
    def optimize(self) -> np.ndarray:
        r"""
        Neumann's algorithm to find the optimal wave function that satisfies all the constraints        
        
        Returns
        ----------
        D: np.ndarray
            Optminal rdm that satisfies the given constraints
        """
        D = np.copy(self.initial_guess)
        norm = []
        for i in range (self.max_iterations):
            for projection in self.constraints:
                D = projection(D)
            norm.append(np.linalg.norm(D - self.initial_guess))
            is_stop = self.alpha * abs(norm[i] - norm[i - 1]) + (1 - self.alpha) * norm[i] < self.eps
            if is_stop:
                break
        return D

class Halpern(Dykstra):
    
    def __init__(self, initial_guess:np.ndarray, constraints:list, alpha:float =1.0, max_iterations:int =100, eps:float =1e-6) -> None:
        super().__init__(initial_guess, constraints,alpha,max_iterations,eps)
        r"""
        Halpern's class for projection onto convex sets.

        Parameters
        ----------
        initial_guess : np.ndarray
            Initial guess for the density matrix \Gamma_0.
        constraints : list
            List of projection operators $J$ onto convex subspaces C_i related to the constraints.
        alpha : float
            Tunning parameter, by default 1
        max_iterations : int, optional
            Number of maximum iterations, by default 100
        eps : float
            Tolerance, by default 1e-6
        """        
    
    def optimize(self) -> np.ndarray:
        r"""
        Halpern's algorithm to find the optimal wave function that satisfies all the constraints        
        
        Returns
        ----------
        gamma_new: np.ndarray
            Optminal rdm that satisfies the given constraints
        """
        gamma_new = np.copy(self.initial_guess)
        norm = []
        for i in range (1,self.max_iterations+1):
            for projection in self.constraints:
                gamma_new = projection(gamma_new)
            gamma_new=(1.0/(i+1))*self.initial_guess +(i/(i+1))*gamma_new
            norm.append(np.linalg.norm(gamma_new - self.initial_guess))
            is_stop = self.alpha * abs(norm[i] - norm[i - 1]) + (1 - self.alpha) * norm[i] < self.eps
            if is_stop:
                break
        return gamma_new

