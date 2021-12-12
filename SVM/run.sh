#!/bin/bash
echo this script should be run from the SVM directory
python SVM_results.py
echo Starting parallel execution of kernel svm
python SVM_kernel_results.py 0.1 &
python SVM_kernel_results.py 0.5 &
python SVM_kernel_results.py 1 &
python SVM_kernel_results.py 5 &
python SVM_kernel_results.py 100 &
