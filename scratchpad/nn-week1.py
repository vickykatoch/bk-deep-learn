import numpy as np
from utils import *
import lr_funcs as lr 


XTrain_Orig, YTrain, XTest_Orig, YTest, classes = load_datasets(False)
XTrain = lr.reshape_features(XTrain_Orig)/255.
XTest = lr.reshape_features(XTest_Orig)/255.


print_stats(XTrain_Orig, YTrain, XTest_Orig, YTest)

d = lr.model(XTrain,YTrain, XTest, YTest, num_iterations=2000 , learning_rate=.005, print_cost=True)


