#!/bin/bash

/usr/bin/python2.7 /home/marshao/SVMClassifier/App_GetPredictData_Fixed_UBT.py
echo "--------------------------------"
echo "Getting data is finished."
echo "--------------------------------"
sleep 2
python /home/marshao/SVMClassifier/App_SVMPredict_Fixed_UBT.py
echo "Prediction is finished."
