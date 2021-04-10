import numpy as np


def f_transition(sigma, TF):
    """
    The function f_transition = Transition_function(rho) provides the value
    of the Transition Function given a value of sigma, s, and q. The definion of
    sigma is:

                pi     rho - R
    sigma  =  ---- . ---------
                2       C  - R

    where R is the rho value that denotes the beginning of the transition region
    and C is the rho value for the end of this region.  Moreover, s and q are
    parameters that characterize the shape of the transition function, and they
    are encapsulated in the object TF.

    In terms of these parameters, the transition function is given by:

                        _                    _                 2     _     _
                    1  |             /  s   |                 q       | \   |
    F(sigma;s,q) = --- |  1  +  tanh|  ---- | tg(sigma) - ----------  |  |  |
                    2  |_            \  pi  |_             tg(sigma) _| /  _|
    """

    tg_sigma = np.tan(sigma)

    thx = np.tanh((TF.s / np.pi) * tg_sigma)
    thy = np.tanh(TF.s * (TF.q ** 2) / np.pi / tg_sigma)

    ff_transition = 0.5 * (1.0 + thx) * (1.0 - thy) / (1.0 - thx * thy)

    return ff_transition


def f_transition_1st(f0, sigma, TF):
    """
    This function provides the value of the first derivative (with respect to sigma!) of the Transition
    Function given a value of the Transition Function itself, and a value of sigma, s, and q.
    The definition of sigma can be seen in the computation of the Transition Function above.
    The parameters s and q characterize the shape of the transition function, and they
    are encapsulated in the object TF.

    In terms of these parameters, the first derivative of the Transition Function with respect to
    sigma is given by:
                                           _                              _
        dF         8s       F (1 - F)     |           2          2         |
     --------  =  ---- . ---------------  |  1  +  ( q  - 1 ) cos (sigma)  |
      dsigma       pi     sin^2(2 sigma)  |_                              _|

    """

    factor1 = 8.0 * TF.s / np.pi

    factor2 = f0 * (1 - f0) / np.sin(2.0 * sigma) / np.sin(2.0 * sigma)

    factor3 = 1.0 + (TF.q ** 2 - 1.0) * np.cos(sigma) * np.cos(sigma)

    ff_transition_1st = factor1 * factor2 * factor3

    return ff_transition_1st


def f_transition_2nd(f0, sigma, TF):
    """
    This function provides the value of the second derivative (with respect to sigma!) of the Transition
    Function given a value of the Transition Function itself, and a value of sigma, s, and q.
    The definition of sigma can be seen in the computation of the Transition Function above.
    The parameters s and q characterize the shape of the transition function, and they
    are encapsulated in the object TF.

    In terms of these parameters, the second derivative of the Transition Function with respect to
    sigma is given by:

                                              _                                           2                                                   _
       d^2 F         32s       F (1 - F)     |   2s          /        2          2       \                    /   4           2    4       \   |
     ----------  =  ----- . ---------------  |  ---- (1-2F) | 1 -  ( q  - 1 ) cos (sigma) |  +  sin(2 sigma) | sin (sigma) - q  cos (sigma) |  |
      dsigma^2        pi     sin^4(2 sigma)  |_  pi          \                           /                    \                            /  _|

    """

    sin_2sigma = np.sin(2.0 * sigma)

    factor1 = 32.0 * TF.s / np.pi

    factor2 = f0 * (1 - f0) / sin_2sigma / sin_2sigma / sin_2sigma / sin_2sigma

    factor3 = (2.0 * TF.s / np.pi) * (1.0 - 2.0 * f0)

    factor4 = 1.0 + (TF.q ** 2 - 1.0) * np.cos(sigma) * np.cos(sigma)

    factor5 = np.sin(sigma) ** 4 - (TF.q ** 2) * (np.cos(sigma) ** 4)

    ff_transition_2nd = (
        factor1 * factor2 * (factor3 * (factor4 ** 2) + sin_2sigma * factor5)
    )

    return ff_transition_2nd
