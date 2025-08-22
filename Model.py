from functools import partial
from deap import creator, tools, algorithms
import deap.base as deap_base
import random, copy
import numpy as np

def make_mutGaussian_clamp(param_ranges):
    def _mut(individual, mu, sigma, indpb):
        # Gaussian variation
        tools.mutGaussian(individual, mu, sigma, indpb)
        # clamp
        for i, (low, high) in enumerate(param_ranges):
            if individual[i] < low:   individual[i] = low
            elif individual[i] > high: individual[i] = high
        return (individual,)
    return _mut

def optimize_pose_GA(eval_fn, param_ranges,
                     ngen=40, mu=60, lambda_=120, cxpb=0.5, mutpb=0.2,
                     sigma=5.0, seed_vec=None, seed_spread=2.0):
    """
    ngen: Number of generations. Run N generations, with each generation undergoing selection, 
        crossover, mutation, and generating a new batch of individuals.
        The larger the number, the longer the computation time, but the more likely it is to find a 
        better solution.
    mu: Population size, i.e., the number of individuals retained in each generation. Here, the 
        eaMuPlusLambda algorithm is used,so mu represents the number of elite individuals. 
        Larger values → cover a broader search space but result in slower computation.
    lambda_: The number of offspring generated per generation. In eaMuPlusLambda: select parent 
        individuals from mu individuals, perform crossover/mutation to generate lambda_ new individuals,
        then select the next generation together with the original mu individuals.
    cxpb: Crossover probability. Determines the probability of two individuals crossing over to generate 
        a new individual. 0.5 indicates a 50% probability of crossover,  and a 50% probability of 
        directly copying the parent.  
    mutpb: Mutation probability. Determines the probability of each individual being mutated after crossover. 
        0.2 = 20% probability of mutation.
    sigma: Standard deviation of mutation (the larger the standard deviation, the greater the mutation 
        amplitude). Here, mutGaussian Gaussian mutation is used, so sigma controls the intensity of 
        random perturbations. For example, if the parameter value is 30 and sigma=5.0, the mutation 
        result might be 30 ± some random numbers.  
    seed_vec: Guidance vector for population initialization (the best solution from the previous stage). 
        If a value is provided, individuals are randomly generated near it (rather than completely 
        randomly). The advantage is a warm start, accelerating convergence.  
    seed_spread: The strength of Gaussian perturbation added near the seed during initialization. 
    Larger values → more dispersed initial population, broader exploration range; smaller values → more 
    concentrated initial population, faster convergence.
    """
    # Defining fitness and individuals
    if 'FitnessMax' in creator.__dict__:
        try:
            del creator.FitnessMax; del creator.Individual
        except Exception:
            pass
    creator.create('FitnessMax', deap_base.Fitness, weights=(1.0,))
    creator.create('Individual', list, fitness=creator.FitnessMax)

    toolbox = deap_base.Toolbox()

    # —— Initialize individuals: Support seed-based bias initialization. ——
    dim = len(param_ranges)
    lows  = np.array([a for a,_ in param_ranges], dtype=float)
    highs = np.array([b for _,b in param_ranges], dtype=float)

    def _rand_ind():
        return [random.uniform(lo, hi) for lo,hi in param_ranges]#我靠 误我好久阿我靠

    def _zero_ind():
        return [0.0 for _ in param_ranges]

    if seed_vec is None:
        init_fn = _zero_ind
    else:
        seed = np.array(seed_vec, dtype=float)
        def _biased():
            cand = seed + np.random.normal(0.0, seed_spread, size=dim)
            cand = np.clip(cand, lows, highs)
            return cand.tolist()
        init_fn = _biased

    toolbox.register('individual', tools.initIterate, creator.Individual, init_fn)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)

    # —— Register operator (note to inject param_ranges into mutate) ——
    toolbox.register('evaluate', eval_fn)
    toolbox.register('mate',     tools.cxBlend, alpha=0.4)
    toolbox.register('mutate',   make_mutGaussian_clamp(param_ranges), mu=0, sigma=sigma, indpb=0.2)
    toolbox.register('select',   tools.selTournament, tournsize=3)

    pop = toolbox.population(mu)
    algorithms.eaMuPlusLambda(pop, toolbox,
                              mu       = mu,
                              lambda_  = lambda_,
                              cxpb     = cxpb,
                              mutpb    = mutpb,
                              ngen     = ngen,
                              verbose  = False)
    best = tools.selBest(pop, k=1)[0]
    return best
