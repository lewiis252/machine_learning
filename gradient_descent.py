import numpy as np
import matplotlib.pyplot as plt

'''gradient descent for f(x,y) = x^2 + y^2'''

# p_0 is starting point
# n_iter is numbers of iters before halt algorithm, if minimum is found earlier than algorithm will not run full n_iter
# alpha is gradient descent step
# epsilon is condition of finding minimum, stationary test is used

def gradient_descent(n_iter=1000, p_0=(1,1), alpha=0.1, epsilon=0.00001):

    indeks = np.arange(n_iter)
    values = np.zeros(n_iter)
    points = np.zeros(n_iter * 2).reshape(n_iter, 2)
    p_i_1 = p_0

    for i in range(0, n_iter):
        p_i = p_i_1 - (alpha * (np.array([2 * p_i_1[0], 2 * p_i_1[1]])))  # alpha * gradient
        p_i_1 = p_i

        points[i] = p_i
        fun = ((p_i[0]) ** 2) + ((p_i[1]) ** 2) # value of function at new point
        values[i] = fun
        max_iter = i # track current iteration

        stop = np.linalg.norm([2 * p_i[0], 2 * p_i[1]]) # check for stop condition (epsilon)

        minimum_found = False
        if stop < epsilon:
            minimum_found = True
            print('Minimum found:', p_i)
            break

    if not minimum_found:
        close_minimum = np.min(values)
        close_minimum_ind = np.where(values == close_minimum)
        print(close_minimum_ind)
        print('There is no minimum that meet epsilon condition, the closest was:', close_minimum, ' at point ',
              points[close_minimum_ind])

    plt.scatter(indeks[:max_iter], values[:max_iter])
    plt.xlabel('iteration')
    plt.ylabel('function value')
    plt.show()



gradient_descent(epsilon=0.00000000001, n_iter=100000, alpha=0.01, p_0=[2,1])



