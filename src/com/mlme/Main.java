/* Author: Daniel Rogers
   Date: 10/16/2019
   Description: This command line program models a dynamic network of sigmoid neurons. The network can be trained using
                back-propagation and stochastic gradient descent (other algorithms will be available in the future)
                over some training set, providing accuracy results along the way. Extra functionality includes saving
                a network to a file, loading in a previously saved networks, visualizing data, displaying network stats.

*/

package com.mlme;

import java.io.*;
import java.util.*;


public class Main {
    // Create constants used to initialize the neural net, and file names.

    private final static int[] LAYER_SIZES = new int[]{784,30,10};
    private final static String TRAINING_FILE = "mnist_train.csv";
    private final static String TESTING_FILE = "mnist_test.csv";
    private final static String PRETRAINED_NET_FILE = "previously_trained_nn.txt";
    private final static String SAVE_FILE = "saved_nn.txt";

    // Converts the provided CSVs (should be in same directory) to a list of double-arrays.
    private static List<double[]> convertDataToDoubleArrays(String filename) throws FileNotFoundException, NumberFormatException{
        Scanner sc = new Scanner(new File(filename));
        List<double[]> training_data = new ArrayList<>();

        // Go through every line of the CSV, splits on "," and parses the elements to doubles
        while (sc.hasNextLine())
        {
            String line = sc.nextLine();
            // line: "0,1,0,1"
            double[] curr_line  = Arrays.stream(line.split(",")).mapToDouble(Double::parseDouble).toArray();
            // curr_line: [0.0, 1.0, 0.0, 1.0]
            training_data.add(curr_line);
        }
        return training_data;
        // training_data: List([0.0, 1.0, 0.0, 1.0],
        //                     [1.0, 0.0, 1.0, 0.0],
        //                     [0.0, 0.0, 1.0, 1.0])
    }

    // Provide printable menu for the user
    private static void printMenu(){
        System.out.println("[1] Train network");
        System.out.println("[2] Load pre-trained network");
        //System.out.println("(the following options are only available after [1] or [2])");
        System.out.println("[3] Display current network accuracy on training data");
        System.out.println("[4] Display current network's accuracy on testing data");
        System.out.println("[5] Save network state to file");
        System.out.println("[6] Run the network on the testing data one case at a time (with visuals)");
        System.out.println("[7] Same as [6], but only show network's misclassifications");
        System.out.println("[8] Display current network's biases and weights");
        System.out.println("[0] Exit program");
    }

    public static void main(String[] args) throws FileNotFoundException, NumberFormatException, IOException{

        boolean exit = false; // variable changed when user enters "0"
        boolean network_trained = false; // keeps track if there's a trained network at play

        Scanner keyboard = new Scanner(System.in);

        List<double[]> training_data = convertDataToDoubleArrays(TRAINING_FILE);
        List<double[]> testing_data = convertDataToDoubleArrays(TESTING_FILE);

        NeuralNet nn = new NeuralNet(LAYER_SIZES);
        StochasticGradientDescent SGD_trainer = new StochasticGradientDescent(training_data);

        System.out.println();
        System.out.println(
                "This is a dynamic-layer 784 input, 10 output network of sigmoid neurons with a focus on the MNIST dataset.\n" +
                "The network is trained using back-propagation and stochastic gradient descent over a training set of\n" +
                "60,000 28x28 images, which are divided into 6000 mini-batches (each batch contains 10 images).");

        // drive user interface
        while (!exit) {
            System.out.println();
            printMenu();
            String input = keyboard.nextLine();

            // stay here until user gets a trained network
            if(!network_trained) {
                System.out.println();
                if ("0".equals(input)) {
                    System.out.println("exiting program...");
                    exit = true;
                } else if ("1".equals(input)) {
                    SGD_trainer.train(nn);
                    network_trained = true;
                } else if ("2".equals(input)) {
                    System.out.println("Loading in pre-trained network...");
                    nn.loadTrainedNeuralNet(PRETRAINED_NET_FILE);
                    System.out.println("... successfully loaded.");
                    network_trained = true;
                } else if ("3".equals(input) || "4".equals(input) || "5".equals(input) || "6".equals(input) || "7".equals(input)) {
                    System.out.println("This option cannot be selected until after [1] or [2].");
                } else if ("8".equals(input)) {
                    System.out.println("(" + (nn.node_layers.get(0).length + nn.node_layers.get(1).length + nn.node_layers.get(2).length) + ") BIASES:");
                    nn.printBiases();
                    nn.printWeightMatrices();
                } else {
                    System.out.println("Invalid input, try again...");
                }
            }

            // once a trained network is at play (ie. network_trained == true), stay here
            else {
                System.out.println();
                if ("0".equals(input)) {
                    System.out.println("exiting program...");
                    exit = true;
                } else if ("1".equals(input)) {
                    SGD_trainer.train(nn);
                } else if ("2".equals(input)) {
                    System.out.println("Loading in pre-trained network...");
                    nn.loadTrainedNeuralNet(PRETRAINED_NET_FILE);
                    System.out.println("... successfully loaded.");
                } else if ("3".equals(input)) {
                    System.out.println("Displaying accuracy on TRAINING data...");
                    nn.printMNISTAccuracy(training_data, "sigmoid");
                } else if ("4".equals(input)) {
                    System.out.println("Displaying accuracy on TESTING data...");
                    nn.printMNISTAccuracy(testing_data, "sigmoid");
                } else if ("5".equals(input)) {
                    nn.saveNNToFile(SAVE_FILE);
                    System.out.println("Saved network to file: " + SAVE_FILE);
                } else if ("6".equals(input)) { // step through testing data, visualizing the data
                    nn.printMNISTAccuracyWithVisual(testing_data, "sigmoid", true);
                } else if ("7".equals(input)) { // visualize misclassifications only
                    nn.printMNISTAccuracyWithVisual(testing_data, "sigmoid", false);
                } else if ("8".equals(input)) {
                    System.out.println("(" + (nn.node_layers.get(0).length + nn.node_layers.get(1).length + nn.node_layers.get(2).length) + ") BIASES:");
                    nn.printBiases();
                    nn.printWeightMatrices();
                } else {
                    System.out.println("Invalid input, try again...");
                }
            }
        }
        keyboard.close();
    }

}
