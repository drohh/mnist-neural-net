# mnist-neural-net
A neural network that learns to classify handwritten digits via the popular MNIST dataset

## Required Project Directory Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~
mnist_train.csv
mnist_test.csv
previously_trained_nn.txt
gson-2.8.6.jar (https://github.com/google/gson)
README.md
src/
  |--com
      |--mlme
          |-- Main.java
          |-- NeuralNet.java
          |-- Node.java
          |-- StochasticGradientDescent.java
~~~~~~~~~~~~~~~~~~~~~~~~~~~

This program makes use of an external jar, GSON, created by Google, that is freely available on their Github. Link provided above.

1.) Compile the java files:
javac -cp ./gson-2.8.6.jar -d "classes" src/com/mlme/*.java

2.) Run the program:
java -cp ".:./gson-2.8.6.jar:classes" com.mlme.Main

*Note: If on Windows, change the “:” in the java command to a “;”*
