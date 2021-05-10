"""
Animated example of Linear Regression with Gradient Descent 

MIT License

Copyright (c) 2021 Luiz Gustavo da Rocha Charamba

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import animation

# cost function method
def chi_square(m, c, X, Y):
    error = 0
    n = len(X)
    for xi, yi in zip(X,Y):
        error += (yi - (m*xi + c))*(yi - (m*xi + c))
    
    return error/n

# partial derivative of chi square for m line parameter
def dChi_dm(m, c, X, Y):
    m_grad = 0
    n = len(X)
    for xi, yi in zip(X,Y):
        m_grad += xi*(yi - (m*xi + c))
    
    return (-2.0/n)*m_grad

# partial derivative of chi square for c line parameter
def dChi_dc(m, c, X, Y):
    c_grad = 0
    n = len(X)
    for xi, yi in zip(X,Y):
        c_grad += (yi - (m*xi + c))
    
    return (-2.0/n)*c_grad

# gradient descent updates line parameters
def grad_descent(m, c, lr, X, Y):
    m_ = m - lr*dChi_dm(m, c, X, Y)
    c_ = c - lr*dChi_dc(m, c, X, Y)
    return m_, c_

def calc_line(m, c, X):
    Y = []
    for xi in X:
        Y.append(m*xi + c)
    return Y

def generate_noisydata(m, c, X):
    noise = np.random.randint(40, size=(50))
    noise = noise - 20
    noise = 0.1*noise
    Y = calc_line(m, c, X)
    X_data = X
    Y_data = Y + noise
    return X_data, Y_data


# original line
m = 0.0
c = 0.0

fig = plt.figure()
ax = plt.axes(xlim=(0, 50), ylim=(0, 30))
data_points, = ax.plot([],[], 'o', color='orange')
line_fitted, = ax.plot([],[])
text = ax.text(0.5, 0.5, '', fontsize=12)

X_data, Y_data = generate_noisydata(0.5, 10, range(0,50))
X_line = X_data
iterations = 5000 #10000
lr = 0.001 # learning rate

# Training
M, C = [m],[c]
for it in range(iterations):
    m,c = grad_descent(m, c, lr, X_data, Y_data)
    M.append(m)
    C.append(c)

def init_animation():
    line_fitted.set_data([], [])
    data_points.set_data(X_data, Y_data)
    return data_points, line_fitted, text

font0 = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 16,
        }

font1 = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }

font2 = {'family': 'serif',
        'color':  'blue',
        'weight': 'normal',
        'size': 14,
        }


def animate(i):
    plt.title('Linear Regression with Gradient Descent optimization in MSE', fontdict=font1)
    mserror = chi_square(M[i], C[i], X_data, Y_data)
    text0 = ax.text(0.5, 28, "Iteration: " + str(i+1), fontdict=font0)
    text1 = ax.text(0.5, 26, "MSE = " + "{:.4f}".format(mserror), fontdict=font1)
    text2 = ax.text(0.5, 23, "Line Parameters: \n" + "(m = " + "{:.4f}".format(M[i]) + ", c = " + "{:.4f}".format(C[i]) + ")", fontdict=font2)

    
    Y_line = calc_line(M[i], C[i], X_line)
    line_fitted.set_data(X_line, Y_line)
    data_points.set_data(X_data, Y_data)
        
    return data_points, line_fitted, text0, text1, text2


print("Line Parameters:\n(m = " + str(m) + ", c = " + str(c) + ")")

# Animation
anim = animation.FuncAnimation(fig, animate, init_func=init_animation,
                               frames=iterations, interval=2, blit=True, repeat=False)

plt.show()
