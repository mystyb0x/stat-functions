import math

def mean_binomial_dist(p: float, n: int) -> float:
    """Calculates the mean of a binomial distribution.
    
    :param p: The p-value.
    :type p: float
    :param n: The sample size. Must be between 0 and 1.
    :type n: int
    :return: The mean of the binomial distribution.
    :rtype: float
    :raises ValueError: If p does not agree with 0 <= p <= 1.
    :raises ValueError: If n is not a non-zero positive integer.
    """
    if not (0 <= p <= 1):
        raise ValueError("p must agree with 0 <= p <= 1.")
    
    if not isinstance(n, int):
        raise ValueError("n must be a non-zero positive integer.")
    
    if n <= 0:
        raise ValueError("n must be a non-zero positive integer.")
        
    return n * p

def mean_geometric_dist(p: float, support = 0) -> float:
    """Calculates the mean of a geometric distribution.
    
    :param p: The p-value.
    :type p: float
    :param support: This represents the two parameterizations of a geometrical distribution.
        0, the default, represents the number of trials neeeded to get one success. (X = 1, 2, 3, ...)
        1 represents the number of failures before the first success. (X = 0, 1, 2, ...)
    :type support: int
    :return: The mean of the geometrical distribution.
    :rtype: float
    :raises ValueError: if p does not agree with 0 < p <= 1.
    :raises ValueError: if support is not 0 or 1. 
    """
    if not (0 < p <= 1):
        raise ValueError("p must agree with 0 < p <= 1.")
    
    if support == 0:
        return 1/p
    elif support == 1:
        return (1-p)/1
    else:
        raise ValueError("support must be 0 or 1.")

def mean_uniform_dist(a: int | float, b: int | float) -> float:
    """Calculates the mean of a uniform distribution.

    :param a: The lower bound.
    :type a: int | float
    :param b: The upper bound.
    :type b: int | float
    :return: The mean of the uniform distribution.
    :rtype: float
    :raises ValueError: if a and b do not agree with a < b.
    """
    if a > b:
        raise ValueError("a and b must agree with a < b.")
    
    return (a * b) / 2

def mean_exponential_dist(λ: int | float) -> float:
    """Calculates the mean of an exponential distribution.
    
    :param λ: The rate parameter. 
    :type λ: int | float
    :return: The mean of the exponential distribution.
    :rtype: float
    :raises ValueError: if λ is less than or equal to 0.
    """
    if λ <= 0:
        raise ValueError("λ cannot be less than or equal to 0.")
    
    return 1 / λ

def stdev_binomial_dist(p: float, n: int) -> float:
    """Calculates the standard deviation of a binomial distribution.

    :param p: The p-value.
    :type p: float
    :param n: The sample size.
    :type n: int
    :return: The standard deviation of the binomial distribution.
    :rtype: float
    :raises ValueError: if p does not agree with 0 <= p <= 1.
    :raises ValueError: if n is not a non-zero positive integer.
    """
    if not (0 <= p <= 1):
        raise ValueError("p must agree with 0 <= p <= 1.")
    
    if not isinstance(n, int):
        raise ValueError("n must be a non-zero positive integer.")
    
    if n <= 0:
        raise ValueError("n must be a non-zero positive integer.")
    
    var = n * p * (1-p)
    return math.sqrt(var)

def stdev_geometric_dist(p: float) -> float:
    """Calculates the standard deviation of a geometric distribution.
    
    :param p: The p-value.
    :type p: float
    :return: The standard deviation of the geometric distribution.
    :rtype: float
    :raises ValueError: if p is not between 0 < p <= 1.
    """
    if not (0 < p <= 1):
        raise ValueError("p must agree with 0 < p <= 1.")
    
    var = (1-p) / (p ** 2)
    return math.sqrt(var)

def stdev_poisson_dist(λ: int | float) -> float:
    """Calculates the standard deviation of a poisson distribution.

    :param λ: The mean.
    :type λ: int | float
    :return: The standard deviation of the poisson distribution.
    :rtype: float
    :raises ValueError: if λ is less than or equal to 0.
    """
    if λ <= 0:
        raise ValueError("λ cannot be less than or equal to 0.")
    
    return math.sqrt(λ)

def stdev_uniform_dist(a: int | float, b: int | float) -> float:
    """Calculates the standard deviation of a uniform distribution.

    :param a: The lower bound
    :type a: int | float
    :param b: The upper bound
    :type b: int | float
    :return: The standard deviation of a uniform distribution.
    :rtype: float
    :raises ValueError: if a and b do not agree with a < b.
    """
    if a > b:
        raise ValueError("a and b must agree with a < b.")
    
    var = (1/12) * ((b - a) ** 2)
    return math.sqrt(var)

def stdev_exponential_dist(λ: int | float) -> float:
    """Calculates the standard deviation of a exponential distribution.

    :param λ: The decay meter
    :type λ: int | float
    :return: The standard deviation of the exponential distribution.
    :rtype: float
    :raises ValueError: if λ is less than or equal to 0.
    """
    if λ <= 0:
        raise ValueError("λ cannot be less than or equal to 0.")
    
    var = 1 / (λ ** 2)
    return math.sqrt(var)