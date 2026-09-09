import numpy as np
def fit_linear(x,y):
    ones=np.ones((x.shape[0],1))
    x_b=np.hstack((ones,x))
    theta=np.linalg.inv(x_b.T.dot(x_b)).dot(x_b.T.dot(y))
    b=theta[0]
    w=theta[1:]
    return b,w
def predict(x,weight,bias):
    return np.dot(x,weight)+bias
x_train = np.array([[3, 10],[1,7],[4, 20]])
y_train = np.array([300, 210, 410])
b,w=fit_linear(x_train,y_train)
x_new=np.array([5,25])
print(predict(x_new,w,b))