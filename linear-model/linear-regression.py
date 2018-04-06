import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, discriminant_analysis, cross_validation


def load_data():
    diabetes = datasets.load_diabetes()
    return cross_validation.train_test_split(diabetes.data, diabetes.target, test_size=0.25, random_state=0)

def test_LinearRegression(*data):
    x_train, x_test, y_train, y_test = data
    reg = linear_model.LinearRegression()
    reg.fit(x_train, y_train)
    print("coefficients:%s\tintercept:%.2f" %(reg.coef_, reg.intercept_))
    print("residual sum of squares:%.2f"%(np.mean(reg.predict(x_test) - y_test) ** 2))
    print("score:%.2f" %(reg.score(x_test, y_test)))
    plt.plot(reg.predict(x_test))
    plt.plot(y_test)
    plt.show()

def test_Ridge(*data):
    x_train, x_test, y_train, y_test = data
    reg = linear_model.Ridge()
    reg.fit(x_train, y_train)
    print("coefficients:%s\tintercept:%.2f" %(reg.coef_, reg.intercept_))
    print("residual sum of squares:%.2f"%(np.mean(reg.predict(x_test) - y_test) ** 2))
    print("score:%.2f" %(reg.score(x_test, y_test)))
    plt.plot(reg.predict(x_test))
    plt.plot(y_test)
    plt.show()

def test_Ridge_alpha(*data):
    x_train, x_test, y_train, y_test = data
    alphas = list(np.arange(1, 100) * 0.01) +list( np.arange(1, 10) * 5)
    scores = []
    for i, alpha in enumerate(alphas):
        reg = linear_model.Ridge(alpha=alpha)
        reg.fit(x_train, y_train)
        scores += [reg.score(x_test, y_test)]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(alphas, scores)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("score")
    ax.set_xscale('log')
    ax.set_title("ridge")
    plt.show()


x_train, x_test,y_train, y_test = load_data()
test_LinearRegression(x_train, x_test, y_train, y_test)   
test_Ridge(x_train, x_test, y_train, y_test)
test_Ridge_alpha(x_train, x_test, y_train, y_test)
