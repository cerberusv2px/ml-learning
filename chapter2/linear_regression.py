from __future__ import division

from sklearn.linear_model import LinearRegression
import numpy as np

X = [[6], [8], [10], [14], [18]]
y = [[7], [9], [13], [17.5], [18]]

model = LinearRegression()
model.fit(X, y)

residual = np.mean((model.predict(X) - y) ** 2)

meanX = np.mean(X)
sumX = 0
xList = []
yList = []
for i in X:
    xList.append(i[0])
    sumX += (i[0] - meanX) ** 2

for i in y:
    yList.append(i[0])

varX = sumX / (len(X) - 1)

numpyVarX = np.var(xList, ddof=1)
numpyCovXY = np.cov(xList, yList)[0][1]
beta = numpyCovXY / numpyVarX
alpha = np.mean(yList) - (beta * np.mean(xList))

print('Variance from numpy: %.2f' % numpyVarX)
print('Co variable from numpy: %.2f' % numpyCovXY)
print('Beta: %.2f' % beta)
print('Alpha: %.2f' % alpha)

print('A 12" pizza cost: $%.2f' % model.predict([15][0]))
print('Residual sum of square: %.2f' % residual)
