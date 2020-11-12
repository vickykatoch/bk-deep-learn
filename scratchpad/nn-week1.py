import numpy as np
import matplotlib.pyplot as plt
from utils import *
import lr_funcs as lr 


XTrain_Orig, YTrain, XTest_Orig, YTest, classes = load_datasets(False)
XTrain = lr.reshape_features(XTrain_Orig)/255.
XTest = lr.reshape_features(XTest_Orig)/255.


# print_stats(XTrain_Orig, YTrain, XTest_Orig, YTest)

def test_learning_rates():
    learning_rates = [0.01, 0.001, 0.0001,.005]
    for i in (learning_rates):
        x = lr.model(XTrain,YTrain, XTest, YTest, num_iterations=5000 , learning_rate=i, print_cost=False)
        plt.plot(np.squeeze(x['costs']), label=str(i))
        print(f'Learning Rate : {i}, Train Accuracy : {x["train_acc"]}, Test Accuracy : {x["test_acc"]}')
    plt.ylabel('cost')
    plt.xlabel('iterations (hundreds)')

    legend = plt.legend(loc='upper center', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    plt.show()


# d = lr.model(XTrain,YTrain, XTest, YTest, num_iterations=2000 , learning_rate=.005, print_cost=True)

# costs = np.squeeze(d['costs'])
# plt.plot(costs)
# plt.ylabel('Cost')
# plt.xlabel('Iterations [per 100')
# plt.title(f'Learning Rate : {d["learning_rate"]}, # of iterations : {d["num_iterations"]}')
# plt.show()

test_learning_rates()