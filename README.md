UFLDL-Tutorial-Solutions
========================

My Matlab code solutions to the famous UFLDL Tutorial:

http://ufldl.stanford.edu/tutorial/selftaughtlearning/SelfTaughtLearning/

(The solutions were actually done when the old websites were available: http://ufldl.stanford.edu/wiki/index.php/UFLDL_Tutorial , http://ufldl.stanford.edu/tutorial/index.php/UFLDL_Tutorial)

As far as I have tested my solutions, it worked as expected in the exercises (see Notes for some clarifications). If you find any bug or have a better solutions in terms of accuracy/efficiency, you are welcome to contact me and give you credit for the changes: paworkprog@gmail.com


Additional remarks:

-I have additionally included in the file "orthonormalICACost.m" of the exercise "Independent component analysis", the
cost and gradient of the same problem stated as a RICA (Reconstruction ICA) if you want to make comparations. 
-All solutions are fully vectorized.


Notes:

-The solution of the exercise "Sparse coding" gets to learn the features, but in a longer number of iterations. Furthermore, I changed the "alpha" parameter, so it to converges faster. There seems to be a bug in the exercise setting, because solutions tend to be not very sparse (and objective funtion increases), so please, feel free to discuss or contact me if you have a better working solution.
-In the exercise about "Convolutional Neural Networks", I just renamed the function files about convolution and pooling so they aren't confused with the ones in the exercise "Convolution and Pooling". The file "cnnExercise" was changed to deal with this.
-The file "sampleIMAGES"in the exercise "Sparse Coding" was renamed to "sampleIMAGES_sp" and changed, so it doesn't get confused with the one of the exercise "Sparse Autoencoder". The file "sparseCodingExercise" was changed to deal with this.
-In much of the solutions, specially when computing gradients, you will find some commented lines: those are wrongly
derived solutions that I just commented so you can see that if you got that answer, it's wrong.
