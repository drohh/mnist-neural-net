/* Author: Daniel Rogers
   Date: 10/16/2019
*/

package com.mlme;

import com.google.gson.Gson;

import java.io.*;
import java.util.*;
import com.google.gson.GsonBuilder;


public class NeuralNet {
    // Initialize nn's attributes
    public List<Node[]> node_layers = new ArrayList<>();
    public List<double[][]> weight_matrices = new ArrayList<>();
    private int input_size = 0;
    private int output_size = 0;

    // Constructor requires an array of the nn's layers' sizes
    public NeuralNet(int[] layer_sizes){
        //System.out.print("Creating a " + Arrays.toString(layer_sizes) + " neural net... ");

        this.input_size = layer_sizes[0];
        this.output_size = layer_sizes[layer_sizes.length-1];

        // Initialize all nodes (and node layers) by looking at each LAYER_SIZES integer
        for(int i = 0; i < layer_sizes.length; ++i){
            Node node_layer[] = new Node[layer_sizes[i]];
            for(int j = 0; j < layer_sizes[i]; ++j){
                node_layer[j] = new Node();
            }
            this.node_layers.add(node_layer);
        }

        // Create weight matrices
        // for an n-layer neural net, there will be n-1 weight matrices, 1 between each node layer
        // each weight matrix's dimensions will be [size of node layer in front] by [size of node layer behind]
        for(int i = 0; i < this.node_layers.size() - 1; ++i){
            double weight_matrix[][] = new double[this.node_layers.get(i+1).length][this.node_layers.get(i).length];
            this.weight_matrices.add(weight_matrix);
        }

        //System.out.println("done.\n");
        //System.out.println("TOTAL WEIGHT MATRICES: " + Integer.toString(weight_matrices.size()));
    }

    // Getters and Setters
    public int getInputSize() {
        return input_size;
    }
    public int getOutputSize() {
        return output_size;
    }
    public List<double[][]> getWeightMatrices() {
        return weight_matrices;
    }
    public void setWeightMatrices(List<double[][]> weight_matrices) {
        this.weight_matrices = weight_matrices;
    }

    // assign the activations of the neural net's first node layer (input layer)
    public void activateInputs(double[] input_array){
        // input_array is 1 element longer than NN's input layer (1st element of input_array is the label)
        // Set input layer's ith node = input_array[i+1]
        for(int i = 0; i < input_array.length-1; ++i){
            this.node_layers.get(0)[i].setActivation(input_array[i+1]);
        }

    }

    // feed activations all the way through the neural net
    public void feedForward(String activation_function){
        if(activation_function == "sigmoid") {
            // no need to calculate activations for input layer, so index from 1 to n-1
            for (int i = 1; i < node_layers.size(); ++i) {
                calculateSigmoidActivations(i);
            }
        }
        //else if different activation functions in future ?
    }

    // function used by feedForward to calculate the activations at each layer
    private void calculateSigmoidActivations(int node_layer_index){
        if(node_layer_index == 0){
            System.out.println("ERROR: Activations for the input layer are not calculated.\n" +
                    "Use the 'activateInputs' method.");
        }
        Node[] node_layer = this.node_layers.get(node_layer_index); // current node layer
        Node[] layer_before = this.node_layers.get(node_layer_index-1); // node layer before this one
        double[][] weight_matrix = this.weight_matrices.get(node_layer_index-1); // current weight matrix

        double[] Z = new double[node_layer.length]; // store Z value of each neuron/node

        for(int i = 0; i < weight_matrix.length; ++i){
            double wx_sum = 0.0;
            for(int j = 0; j < weight_matrix[i].length; ++j){
                //System.out.println(weight_matrix[i][j] + "*" + layer_before[j].getActivation());
                wx_sum += (weight_matrix[i][j] * layer_before[j].getActivation()); // dot product of weight matrix and activation's of layer before
            }
            //System.out.println("Bias: " + node_layer[i].getBias());
            // Z = Weights&LayerBeforeActivations summation + biases of current layer's nodes
            Z[i] = wx_sum + node_layer[i].getBias();
            //System.out.println("Z: " + Z[i]);
            // activation = sigma(Z) where Z is WX+B
            node_layer[i].setActivation(sigma(Z[i]));
        }

    }

    // simple sigma utility
    private double sigma(double z){
        return 1.0/(1.0+Math.exp(-z));
    }

    // return the index of the node with the highest activation in the neural net's last layer
    public int returnHighestOutputIndex(){
        int largest = 0; // first index is the largest so far
        Node[] output_layer = this.node_layers.get(node_layers.size() - 1); // grab last layer
        for ( int i = 1; i < output_layer.length; i++ ) // go through every node
        {
            // if node's activation is larger than the activation at the 'largest' index, update 'largest'
            if ( output_layer[i].getActivation() > output_layer[largest].getActivation() ) largest = i;
        }
        return largest;
    }

    // utility for going through each weight matrix of the neural net and assigning random values to its elements
    public void randomizeWeightMatrices(){
        Random rng = new Random();
        for(int i = 0; i < this.weight_matrices.size(); ++i){ // each weight matrix (i)
            for(int j = 0; j < this.weight_matrices.get(i).length; ++j){ // each array (j) in the weight matrix i
                for(int k = 0; k < this.weight_matrices.get(i)[j].length; ++k){ // each element (k) in array j
                    //this.weight_matrices.get(i)[j][k] = Math.random(); // Random number from 0 to 1
                    this.weight_matrices.get(i)[j][k] = rng.nextDouble() * 2 - 1; // Random number from -1 to 1
                }
            }
        }
    }

    // print activations of a particular node layer
    public void printActivations(int node_layer_index){
        System.out.println("Activations for Layer #" + node_layer_index + ":");
        for(Node node : this.node_layers.get(node_layer_index)){ // for every node in this node layer
            System.out.print(node.getActivation());
            System.out.print(" ");
        }
        //Print two new lines, because the loop above uses 'print' instead of 'println'
        System.out.println("");
        System.out.println("");
    }

    // print activations of all node layers
    public void printAllActivations(){
        for(int i = 0; i < this.node_layers.size(); ++i){
            printActivations(i);
        }
    }

    // print weight matrices along with dimensionality info
    public void printWeightMatrices(){
        System.out.println("WEIGHT MATRICES DIMENSIONALITY:");
        for(int i = 0; i < this.weight_matrices.size(); ++i){
            System.out.println("Layer" + Integer.toString(i) + "->Layer" + Integer.toString(i+1) + "\t" +
                   Integer.toString(weight_matrices.get(i).length) + "x" + Integer.toString(weight_matrices.get(i)[0].length));
            System.out.println(Arrays.deepToString(weight_matrices.get(i)));

        }
        System.out.println();
    }

    // print biases for all nodes
    public void printBiases(){
        for(Node[] layer : this.node_layers){
            for(Node node : layer){
                System.out.print(node.getBias() + " ");
            }
        }
        System.out.println();
    }

    // print accuracy of the network using this activation function on this MNIST data
    public void printMNISTAccuracy(List<double[]> data, String activation_function){
        int total_correct = 0; // running total of times the network has been correct
        int[] nn_correct_nums = new int[10]; // running count of the specific labels/numbers the network was correct on
        int[] num_instances = new int[10]; // instances of every label/number, used to calculate accuracy of network on each particular label

        for(double[] data_case : data){ // go through every input of the data set
            num_instances[(int)data_case[0]] += 1; // add 1 to the label's instances count

            // create a hot vector, where the "1" is on the index of the input's label
            double[] hot_vector = new double[10];
            hot_vector[(int)data_case[0]] = 1;

            // activate and feed the case through the network
            this.activateInputs(data_case);
            this.feedForward(activation_function);

            // check to see if network was right
            if(data_case[0] == this.returnHighestOutputIndex()){
                // if so, increase that number's count in nn_correct_nums, and add 1 to total_correct
                nn_correct_nums[this.returnHighestOutputIndex()] += 1;
                total_correct += 1;
                //System.out.println("Correct!");
            }
        }

        float accuracy = ((float)total_correct / (float)data.size())*100; // calculate accuracy percentage

        // Pretty print the stats
        System.out.println("Network Accuracy: " + "\n----------");
        System.out.println("0: " + nn_correct_nums[0] + "/" + num_instances[0] + "  1: " + nn_correct_nums[1] + "/" + num_instances[1] +
                "  2: " + nn_correct_nums[2] + "/" + num_instances[2] + "  3: " + nn_correct_nums[3] + "/" + num_instances[3] + "  4: " + nn_correct_nums[4] + "/" + num_instances[4]);
        System.out.println("5: " + nn_correct_nums[5] + "/" + num_instances[5] + "  6: " + nn_correct_nums[6] + "/" + num_instances[6] +
                "  7: " + nn_correct_nums[7] + "/" + num_instances[7] + "  8: " + nn_correct_nums[8] + "/" + num_instances[8] + "  9: " + nn_correct_nums[9] + "/" + num_instances[9]);
        System.out.println("Total Accuracy: " + total_correct + "/" + data.size() + " = " + accuracy + "%");
    }

    // load in network weights and biases from a file
    public void loadTrainedNeuralNet(String nn_file) throws FileNotFoundException{
        Scanner sc = new Scanner(new File(nn_file)); // create a scanner of the file, used to step through each line

        String bias_line = sc.nextLine(); // first line should be list of all the biases of the network, space-separated
        // bias_line: "-0.213 0.998 .453"...
        // split the bias_line on the spaces, then parse each number as a double, assigning all of these to an array
        double[] biases  = Arrays.stream(bias_line.split(" ")).mapToDouble(Double::parseDouble).toArray();

        int bias_index = 0; // keep track of the index you're on as you go through the 'biases' array
        for(int i = 0; i < node_layers.size(); ++i){ // for every node layer
            for(int j = 0; j < node_layers.get(i).length; ++j){ // for every node
                node_layers.get(i)[j].setBias(biases[bias_index]); // change this node's bias to the current 'biases' array element
                ++bias_index; // go to the next number in the 'biases' array
            }
        }

        List<double[][]> weight_matrices = new ArrayList<>(); // create a new list of weight matrices

        // use Gson library to load in the matrices as 2D double-arrays
        Gson gson = new GsonBuilder().create();
        double[][] first_matrix = gson.fromJson(sc.nextLine(), double[][].class);
        double[][] second_matrix = gson.fromJson(sc.nextLine(), double[][].class);

        // add the matrices to new list of weight matrices
        weight_matrices.add(first_matrix);
        weight_matrices.add(second_matrix);

        // set that list as the neural net's new weight matrices
        setWeightMatrices(weight_matrices);

    }

    // print weight matrices, each on 1 line, with no labeling
    public void printSimpleWeightMatrices(){
        for( double[][] weight_matrix : this.getWeightMatrices()){
            System.out.println(Arrays.deepToString(weight_matrix));
        }
    }

    // return a string of a space-separated list of every node's bias
    public String returnBiasesString(){
        String result = "";
        for(Node[] layer : this.node_layers){
            for(Node node : layer){
                result += node.getBias() + " ";
            }
        }
        return result;
    }

    // save the neural net's weights and biases to a file
    public void saveNNToFile(String filename){
        PrintStream stdout = System.out; // save the current stdout print stream for later
        try {
            System.setOut(new PrintStream(filename)); // point stdout print stream to the file
        } catch (Exception e){
            System.err.println("Exception caught: " + e); // catch the IOException
        }
        // now with the stdout print stream directed to the file, use print utilities to print out network's weights and biases
        this.printBiases();
        this.printSimpleWeightMatrices();

        // set the stdout print stream back to what it was before
        System.setOut(stdout);

    }

    // step through MNIST data, printing the input in ASCII art, and comparing the label to the network's classification
    public void printMNISTAccuracyWithVisual(List<double[]> data, String activation_function, boolean display_all) {

        Scanner sc = new Scanner(System.in); // new scanner to read user input at the command line

        for(int i = 0; i < data.size(); ++i) { // for every input in the data
            double[] curr_data = data.get(i); // current input
            boolean correct = false; // keep track if network was correct
            int correct_label = (int) curr_data[0]; // the first element of the data is the correct label
            String userin = "";

            // activate the network with the data and feed it forward
            this.activateInputs(curr_data);
            this.feedForward(activation_function);

            // if the correct label is the same as the network's best guess (highest output in final layer), update 'correct' to true
            if (correct_label == this.returnHighestOutputIndex()) {
                correct = true;
            }
            if (display_all) { // if display_all == true, show ascii art and network classification even if network was correct
                String correct_string = (correct) ? "  Correct!" : "Incorrect!"; // if the network was correct, get ready to display Correct!, else Incorrect!
                System.out.println("Testing Case #" + (i+1) + ":   " + "Correct classification = " + correct_label + "   Network Classification = " + this.returnHighestOutputIndex() + "   " + correct_string);
                printMNISTAscii(curr_data); // visualize the input array
                System.out.println("Enter 1 to continue. All other values return to main menu.");

                userin = sc.nextLine();
		// only continue if user enters a "1" 
		if (!"1".equals(userin)) {
		    break;
		}

            } else { // if display_all == false, only show ascii art and network classification for MISclassifications
                if(!correct){ // if network was incorrect with its guess (ie. returnHighestOutputIndex != input label)
                    System.out.println("Testing Case #" + (i+1) + ":   " + "Correct classification = " + correct_label + "   Network Classification = " + this.returnHighestOutputIndex());
                    printMNISTAscii(curr_data); // visualize
                    System.out.println("Enter 1 to continue. All other values return to main menu.");

                    userin = sc.nextLine();
		    // only continue if user enters a "1" 
		    if (!"1".equals(userin)) {
			break;
		    }
                }
            }

        }

    }

    // print an ascii art drawing of an input line of MNIST data
    private void printMNISTAscii(double[] input){
        // ascii image is 28x28 characters

        for(int i = 0; i < 784; i+=28){ // go through the input data 28 lines at a time (because that's 1 ascii line at a time)
            for(int j = 0; j < 28; ++j){ // cycle through this block of 28
                //assign and print an ascii character corresponding to the value of this element
                System.out.print(returnMNISTAsciiGreyscaleChar(input[i+j]));
            }
            System.out.println(); // after printing 28 ascii art characters, go to the next line
        }
    }

    // utility function to assign a particular MNIST value to an ascii character
    private Character returnMNISTAsciiGreyscaleChar(double num){
        Character result;
        if(num < 25){
            result = ' ';
        } else if(num < 50) {
            result = '.';
        } else if(num < 76) {
            result = '\\';
        } else if(num < 102) {
            result = '=';
        } else if(num < 127) {
            result = ':';
        } else if(num < 152) {
            result = '*';
        } else if(num < 178) {
            result = '|';
        } else if(num < 204) {
            result = '#';
        } else if(num < 230) {
            result = 'M';
        } else{
            result = '&';
        }

        return result;
    }
}
