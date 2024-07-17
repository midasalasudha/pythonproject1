from statistics import mean

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import style


def calculate_slope_intercept(xvalues, yvalues):
    m = (((mean(xvalues) * mean(yvalues)) - mean(xvalues * yvalues)) /
         ((mean(xvalues) * mean(xvalues)) - mean(xvalues * xvalues)))
    b = mean(yvalues) - m * mean(xvalues)
    return m, b


def linear_Regression():
    regression_line = [(m * x) + b for x in xvalues]
    style.use('ggplot')
    plt.title('Training Data & Regression Line')
    plt.scatter(xvalues, yvalues, color='#003F72', label='Training Data')
    plt.plot(xvalues, regression_line, label='Reg Line')
    plt.legend(loc='best')
    plt.show()


def test_data():
    predict_xvalue = 7
    predict_yvalue = (m * predict_xvalue) + b
    print('Test Data for x :     ', predict_xvalue, '    ', 'Test Data for y :     ', predict_yvalue)
    plt.title('Train & Test Value')
    plt.scatter(xvalues, yvalues, color='#003F72', label='data')
    plt.scatter(predict_xvalue, predict_yvalue, color='#ff0000', label='Predicted Value')
    plt.legend(loc='best')
    plt.show()


def validate_results():
    predict_xvalues = np.array([2.5, 3.5, 4.5, 5.5, 6.5], dtype=np.float64)
    predict_yvalues = [(m * x) + b for x in predict_xvalues]
    print('Validation Data Set')
    print('X values', predict_xvalues)
    print('Y values', predict_yvalues)


# driver
xvalues = np.array([1, 2, 3, 4, 5], dtype=np.int32)
yvalues = np.array([14, 24, 34, 44, 54], dtype=np.int32)
m, b = calculate_slope_intercept(xvalues, yvalues)
print('Slope :  ', m, 'Intercept :  ', b)
linear_Regression()
test_data()
validate_results()
