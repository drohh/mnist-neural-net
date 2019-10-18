Project Directory Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~
mnist_train.csv
mnist_test.csv
previously_trained_nn.txt
gson-2.8.6.jar (https://github.com/google/gson)
README.txt
src/
  |--com
      |--mlme
          |-- Main.java
          |-- NeuralNet.java
          |-- Node.java
          |-- StochasticGradientDescent.java
~~~~~~~~~~~~~~~~~~~~~~~~~~~

This program makes use of an external jar, GSON, created by Google, that is freely available on their Github. Link provided above.

1.) Compile the java files (I uses “classes” here as the name of the directory the compiled classes will be stored into):

javac -cp ./gson-2.8.6.jar -d "classes" src/com/mlme/*.java

2.) Run the program, with a class path that includes your current directory for all of the needed txt and csv files, the external jar, and your new classes folder (if you changed the “classes” directory when you compiled, be sure to change “classes” here to reflect that change):

java -cp ".:./gson-2.8.6.jar:classes" com.mlme.Main

*If on Windows, change the “:” in the java command to a “;”
