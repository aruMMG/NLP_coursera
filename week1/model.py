import numpy as np


def sigmoid(z):
    h = 1 / (1 + np.exp(-z))
    return h

def gradientDecent(x, y, theta, alpha, num_iters):
    m = x.shape[0]
    for i in range(0, num_iters):
        z = np.dot(x, theta)
        h = sigmoid(z)
        J = -1/m * (np.dot(y.transpose(), np.log(h)) + np.dot((1-y).transpose(), np.log(1-h)))
        theta = theta - (alpha/m) * np.dot(x.transpose(), (h-y))

    J = float(J)
    return J, theta


if __name__=="__main__":
    np.random.seed(3)
    tem_x = np.append(np.ones((10,1)), np.random.rand(10,2) * 2000, axis=1)
    tem_y= (np.random.rand(10,1)>0.35).astype(float)
    print(tem_x.shape)
    tmp_j, tem_theta = gradientDecent(tem_x, tem_y, np.zeros((3,1)), 1e-8, 700)
    print(f"The cost after training is {tmp_j:.8f}.")
    print(f"The resulting vector of weights is {[round(t, 8) for t in np.squeeze(tem_theta)]}")