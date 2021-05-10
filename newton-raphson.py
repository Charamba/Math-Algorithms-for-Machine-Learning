"""
Example of Newton-Raphson method for finding roots 

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
import pandas as pd
import matplotlib.pyplot as plt 

def f(x):
  return x**6/6 - 3*x**4 - 2*x**3/3 + 27*x**2/2 + 18*x - 30

def d_f(x):
  return x**5 - 12*x**3 -2*x**2 + 27*x + 18

def newton_raphson(x):
    return x - f(x) / d_f(x)

x = -4.5

d = {"x" : [x], "f(x)": [f(x)]}
nr_X = []
nr_Y = []
iterations = 1000
for i in range(0, iterations):
  x = newton_raphson(x)
  d["x"].append(x)
  nr_X.append(x)
  d["f(x)"].append(f(x))
  nr_Y.append(f(x))

print("Iterations: ", len(nr_X))
print(pd.DataFrame(d, columns=['x', 'f(x)']))

X = []
Y = []
left_limit = -5
right_limit = 5
step_sample = 0.01

for x in np.arange(left_limit, right_limit, step_sample):
    y = f(x)
    X.append(x)
    Y.append(y)


plt.plot(X, Y)
plt.plot(nr_X, nr_Y, color='r', marker='o')
plt.plot([left_limit, right_limit], [0,0], color="black")
plt.show()

    

    