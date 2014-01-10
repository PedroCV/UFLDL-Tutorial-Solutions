UFLDL-Tutorial-Solutions
========================

My Matlab code solutions to the famous UFLDL Tutorial. It includes all the proposed exercises from these two lists:

-http://ufldl.stanford.edu/wiki/index.php/UFLDL_Tutorial

-http://ufldl.stanford.edu/tutorial/index.php/UFLDL_Tutorial

As far as I have tested my solutions, it worked as expected in the exercises (see Notes for some clarifications). If you find any bug or have a better solutions in terms of accuracy/efficiency, you are welcome to contact me and give you credit for the changes: paworkprog@gmail.com


Additional remarks:

-I have additionally included in the file "orthonormalICACost.m" of the exercise "Independent component analysis", the
cost and gradient of the same problem stated as a RICA (Reconstruction ICA) if you want to make comparations. 
-All solutions are fully vectorized.


Notes:

-The solution of the exercise "Sparse coding" gets to learn the features, but in a longer number of iterations. Furthermore, I changed the "alpha" parameter, for it to converge faster. There seems to be a bug in the exercise setting, because solutions tend to be not very sparse, so please, feel free to discuss or contact me if you have a better working
solution.
-In much of the solutions, specially when computing gradients, you will find some commented lines: those are wrongly
derived solutions that I just commented so you can see that if you got that answer, it's wrong.
