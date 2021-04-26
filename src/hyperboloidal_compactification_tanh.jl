using Markdown

doc"""
    f_transition(sigma, TF)
Transition Function (rho) given a value of sigma, s, and q.

```
            pi     rho - R
sigma  =  ---- . ---------
            2       C  - R
```

where R is the rho value that denotes the beginning of the transition region
and C is the rho value for the end of this region.  Moreover, s and q are
parameters that characterize the shape of the transition function, and they
are encapsulated in the object TF.

In terms of these parameters, the transition function is given by:

```                 _                    _                 2     _     _
                1  |             /  s   |                 q       | \   |
F(sigma;s,q) = --- |  1  +  tanh|  ---- | tg(sigma) - ----------  |  |  |
                2  |_            \  pi  |_             tg(sigma) _| /  _|
```
"""
function f_transition(sigma, TF)

    tg_sigma = tan(sigma)

    thx = tanh((TF.s / pi) * tg_sigma)
    thy = tanh(TF.s * (TF.q^2) / pi / tg_sigma)

    ff_transition = 0.5 * (1.0 + thx) * (1.0 - thy) / (1.0 - thx * thy)

    return ff_transition
end


doc"""
    f_transition_1st(f0, sigma, TF)
First derivative (with respect to sigma!) of the Transition
The parameters s and q characterize the shape of the transition function

In terms of these parameters, the first derivative of the Transition Function with respect to
sigma is given by:
```                                    _                              _
    dF         8s       F (1 - F)     |           2          2         |
 --------  =  ---- . ---------------  |  1  +  ( q  - 1 ) cos (sigma)  |
  dsigma       pi     sin^2(2 sigma)  |_                              _|
```

"""
function f_transition_1st(f0, sigma, TF)

    factor1 = 8.0 * TF.s / pi

    factor2 = f0 * (1 - f0) / sin(2.0 * sigma) / sin(2.0 * sigma)

    factor3 = 1.0 + (TF.q^2 - 1.0) * cos(sigma) * cos(sigma)

    ff_transition_1st = factor1 * factor2 * factor3

    return ff_transition_1st
end


doc"""
    f_transition_2nd(f0, sigma, TF)
Second derivative (with respect to sigma!) of the Transition Function
The parameters s and q characterize the shape of the transition function

                                          _                                           2                                                   _
   d^2 F         32s       F (1 - F)     |   2s          /        2          2       \                    /   4           2    4       \   |
 ----------  =  ----- . ---------------  |  ---- (1-2F) | 1 -  ( q  - 1 ) cos (sigma) |  +  sin(2 sigma) | sin (sigma) - q  cos (sigma) |  |
  dsigma^2        pi     sin^4(2 sigma)  |_  pi          \                           /                    \                            /  _|

"""
function f_transition_2nd(f0, sigma, TF)

    sin_2sigma = sin(2.0 * sigma)

    factor1 = 32.0 * TF.s / pi

    factor2 = f0 * (1 - f0) / sin_2sigma / sin_2sigma / sin_2sigma / sin_2sigma

    factor3 = (2.0 * TF.s / pi) * (1.0 - 2.0 * f0)

    factor4 = 1.0 + (TF.q^2 - 1.0) * cos(sigma) * cos(sigma)

    factor5 = sin(sigma)^4 - (TF.q^2) * (cos(sigma)^4)


    ff_transition_2nd = (
        factor1 * factor2 * (factor3 * (factor4^2) + sin_2sigma * factor5)
    )

    return ff_transition_2nd
end
