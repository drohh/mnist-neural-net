/* Author: Daniel Rogers
   Date: 10/16/2019
*/

package com.mlme;


import java.util.*;


public class StochasticGradientDescent {

    // set constants for the algorithm
    private final static String ACTIVATION_FUNCTION = "sigmoid";
    private final static int MINI_BATCH_SIZE = 10;
    private final static double LEARNING_RATE = 3.0;
    private final static int NUMBER_OF_EPOCHS = 30;
    private final static double SCALE_FACTOR = 1.0/255.0;
    private final static float ACCURACY_THRESHOLD = 98;

    private List<double[]> training_data = new ArrayList<>();

    // constructor, requires a list of double-array to instantiate
    public StochasticGradientDescent(List<double[]> training_data){
        this.training_data = training_data;
    }

    // create the mini batches for the algorithm will cycle through
    private List<List<double[]>> createRandomMiniBatches(){
        Collections.shuffle(this.training_data); // shuffle the training data
        List<List<double[]>> mini_batches = new ArrayList<>(); // create the big list that will hold all the mini-batches

        // go through the data in blocks of size MINI_BATCH_SIZE
        for (int i = 0; i < this.training_data.size(); i += MINI_BATCH_SIZE) {
            // grab a MINI_BATCH_SIZE amount of input lines from the training data and add them to a list of double-arrays
            List<double[]> curr_mini_batch = new ArrayList<>();
            for (int j = 0; j < MINI_BATCH_SIZE; ++j ){
                curr_mini_batch.add(this.training_data.get(i+j));
            }
            mini_batches.add(curr_mini_batch); // add that list of double-arrays to the big list of mini-batches
        }
        //System.out.println("MINI-BATCH COUNT: " + Integer.toString(mini_batches.size()));
        //System.out.println();
        return mini_batches;
    }

    // when training on MNIST data, pretty print the stats after each epoch
    private void printMNISTEpochStats(int total_correct_in_epoch, int epoch_count, int[] nn_correct_nums, int[] num_instances) {
        float accuracy = ((float)total_correct_in_epoch / (float)this.training_data.size())*100;

        System.out.println("Epoch #" + epoch_count + "\n----------");
        System.out.println("0: " + nn_correct_nums[0] + "/" + num_instances[0] + "  1: " + nn_correct_nums[1] + "/" + num_instances[1] +
                "  2: " + nn_correct_nums[2] + "/" + num_instances[2] + "  3: " + nn_correct_nums[3] + "/" + num_instances[3] + "  4: " + nn_correct_nums[4] + "/" + num_instances[4]);
        System.out.println("5: " + nn_correct_nums[5] + "/" + num_instances[5] + "  6: " + nn_correct_nums[6] + "/" + num_instances[6] +
                "  7: " + nn_correct_nums[7] + "/" + num_instances[7] + "  8: " + nn_correct_nums[8] + "/" + num_instances[8] + "  9: " + nn_correct_nums[9] + "/" + num_instances[9]);
        System.out.println("Total Accuracy: " + total_correct_in_epoch + "/" + this.training_data.size() + " = " + accuracy + "%\n");
    }

    // create a hot vector, where the "1" represents the correct label of the training input line
    private double[] generateMNISTOutputVector(int proper_index){
        double[] hot_vector = new double[10]; // all 0s at first
        hot_vector[proper_index] = 1; // identify the the index of that represents the label in the input line
        return hot_vector;
    }

    // train a neural network
    public void train(NeuralNet nn){
        float accuracy = 0; // accuracy, which will be calculated at the end of every epoch
        int epoch_count = 0; // total count of epochs
        int mini_batch_count = 0; // total count of mini batches

        System.out.println("Training...\n");

        nn.randomizeWeightMatrices();

        // Scale inputs down to 0-1 range
        for(int i=0; i< this.training_data.size(); i++) {
            for(int j = 1; j < this.training_data.get(i).length; ++j){
                this.training_data.get(i)[j] *= SCALE_FACTOR;
            }
        }

        // Start Algorithm Loop
        while(epoch_count != NUMBER_OF_EPOCHS && accuracy < ACCURACY_THRESHOLD) {
            int[] num_instances = new int[10]; // keep track of each label that is seen
            int[] nn_correct_nums = new int[10]; // keep track of neural net's correct guesses
            int total_correct_in_epoch = 0; // count of network's correct classifications per epoch

            // Create list of mini-batches, randomized data each epoch
            List<List<double[]>> mini_batches = createRandomMiniBatches();

            for(List<double[]> mini_batch : mini_batches){
                // Create WEIGHT matrices to keep track of the overall weight gradients from the mini batch
                List<double[][]> weight_gradient_matrices = new ArrayList<>();
                for(int i = 0; i < nn.node_layers.size() - 1; ++i){
                    double weight_gradient_matrix[][] = new double[nn.node_layers.get(i+1).length][nn.node_layers.get(i).length];
                    weight_gradient_matrices.add(weight_gradient_matrix);
                }
                //System.out.println(curr_weight_gradient_matrices.get(0).length + "x" + curr_weight_gradient_matrices.get(0)[0].length);

                // Create BIAS vector to keep track of the overall bias gradients from the mini batch
                List<double[]> bias_gradients = new ArrayList<>();
                for(Node[] node_layer : nn.node_layers){
                    double[] bias_gradient_layer = new double[node_layer.length];
                    bias_gradients.add(bias_gradient_layer);
                }
                //System.out.println(curr_bias_gradients.get(0).length);

                for(double[] training_case : mini_batch){

                    // Create weight gradient matrices for current training case
                    List<double[][]> curr_weight_gradient_matrices = new ArrayList<>();
                    for(int i = 0; i < nn.node_layers.size() - 1; ++i){
                        double weight_gradient_matrix[][] = new double[nn.node_layers.get(i+1).length][nn.node_layers.get(i).length];
                        curr_weight_gradient_matrices.add(weight_gradient_matrix);
                    }

                    // Create bias gradient vectors for current training case
                    List<double[]> curr_bias_gradients = new ArrayList<>();
                    for(Node[] node_layer : nn.node_layers){
                        double[] bias_gradient_layer = new double[node_layer.length];
                        curr_bias_gradients.add(bias_gradient_layer);
                    }

                    num_instances[(int)training_case[0]] += 1; // add 1 to the instances of this label

                    // get a hot vector
                    double[] correct_output = generateMNISTOutputVector((int)training_case[0]);

                    //System.out.println("Training Case = " + Arrays.toString(training_case));
                    //System.out.println("1-Hot Output Vector: " + Arrays.toString(correct_output));

                    // Set input layer activations (first number in training case is the label)
                    nn.activateInputs(training_case);
                    //nn.printActivations(0);

                    // Feed forward through layers
                    //nn.printAllActivations();
                    nn.feedForward(ACTIVATION_FUNCTION);
                    //nn.printAllActivations();

                    if(training_case[0] == nn.returnHighestOutputIndex()){
                        // add 1 to this label's correct count and add 1 to network's overall correct count
                        nn_correct_nums[nn.returnHighestOutputIndex()] += 1;
                        total_correct_in_epoch += 1;
                        //System.out.println("Correct!");
                    }

                    // ~~~~~~~~~~~~~~~~~~~ BACK PROPAGATE ~~~~~~~~~~~~~~~~~~~~

                    // change the last layer's bias gradients
                    Node[] last_layer = nn.node_layers.get(nn.node_layers.size()-1);
                    for(int i = 0; i < last_layer.length; ++i){
                        // bias grad. = (ai - correct_outputi) * ai * (1-ai)
                        double curr_activation = last_layer[i].getActivation();
                        double bg = (curr_activation - correct_output[i]) * curr_activation * (1-curr_activation);
                        //last_layer[i].setBiasGradient(bg);
                        curr_bias_gradients.get(curr_bias_gradients.size()-1)[i] += bg;
                        //System.out.println("BG: " + bg);
                    }

                    // change all intermediate layer's bias gradients
                    for(int i = nn.node_layers.size()-2; i > 0; --i){ // every node layer
                        double[] layer_before_bg = curr_bias_gradients.get(i+1); // Since we're going backwards, the layer before is actually i+1
                        for (int j = 0; j < nn.node_layers.get(i).length; ++j){ // every node
                            double weighted_sum = 0;
//                            System.out.println("Layer" + Integer.toString(i) + "->Layer" + Integer.toString(i+1) + "\t" +
//                                    Integer.toString(nn.weight_matrices.get(i).length) + "x" + Integer.toString(nn.weight_matrices.get(i)[0].length));
                            //System.out.println(Arrays.deepToString(nn.weight_matrices.get(i)));
                            for(int k = 0; k < nn.weight_matrices.get(i).length; ++k){
                                //System.out.println("i: " + i + ", j: " + j + ", k: " + k);
                                //System.out.println(nn.weight_matrices.get(i)[k][j] + " * " + layer_before_bg[k]);
                                weighted_sum += nn.weight_matrices.get(i)[k][j]  * layer_before_bg[k];
                            }
                            //System.out.println("weighted_sum: " + weighted_sum + ", node activation:" + nn.node_layers.get(i)[j].getActivation());
                            //System.out.println("BG: " + weighted_sum * nn.node_layers.get(i)[j].getActivation() * (1 - nn.node_layers.get(i)[j].getActivation()));
                            curr_bias_gradients.get(i)[j] += weighted_sum * nn.node_layers.get(i)[j].getActivation() * (1 - nn.node_layers.get(i)[j].getActivation());
                        }
                        //System.out.println("Finished layer #" + i);
                    }
                    //System.out.println("Done with intermediate layers");

                    // use bias gradients to calculate weight gradients
                    for(int i = 0; i < weight_gradient_matrices.size(); ++i) { // each matrix [][]
                        for(int j = 0; j < weight_gradient_matrices.get(i).length; ++j) { // each row []
                            for(int k = 0; k < weight_gradient_matrices.get(i)[j].length; ++k) { // each element
                                double wg = nn.node_layers.get(i)[k].getActivation() * curr_bias_gradients.get(i + 1)[j];
                                //System.out.println("WG: " + wg);
                                weight_gradient_matrices.get(i)[j][k] += wg;
                            }
                        }
                    }

                    // add this training case's bias gradients to the overall bias gradient tracker
                    for(int i = 0; i < curr_bias_gradients.size(); ++i){
                        for(int j = 0; j < curr_bias_gradients.get(i).length; ++j){
                            bias_gradients.get(i)[j] += curr_bias_gradients.get(i)[j];
                        }
                    }

                    // add this training case's weight gradients to the overall weight gradient tracker
                    for(int i = 0; i < curr_weight_gradient_matrices.size(); ++i) {
                        for (int j = 0; j < curr_weight_gradient_matrices.get(i).length; ++j) {
                            for (int k = 0; k < curr_weight_gradient_matrices.get(i)[j].length; ++k) {
                                weight_gradient_matrices.get(i)[j][k] += curr_weight_gradient_matrices.get(i)[j][k];
                            }
                        }
                    }
                    //~~~~~~~~~~~~~~~~ END OF BACK PROPAGATION ~~~~~~~~~~~~~~~~~
                } // End of Mini-batch

                mini_batch_count += 1;
                //System.out.println("MINI BATCH #" + mini_batch_count + " FINISHED\n");

                // Print overall weight gradients
//                for(int i = 0; i < weight_gradient_matrices.size(); ++i){
//                    System.out.println(Arrays.deepToString(weight_gradient_matrices.get(i)));
//                }

                // Revise weights
                for(int i = 0; i < nn.weight_matrices.size(); ++i){
                    for(int j = 0; j < nn.weight_matrices.get(i).length; ++j){
                        for(int k = 0; k < nn.weight_matrices.get(i)[j].length; ++k){
                            double old_weight = nn.weight_matrices.get(i)[j][k];
                            //System.out.println("Old weight: " + old_weight);
                            //System.out.println(nn.weight_matrices.get(i)[j][k] = old_weight - ((double)LEARNING_RATE)/((double)MINI_BATCH_SIZE) * weight_gradient_matrices.get(i)[j][k]);
                            nn.weight_matrices.get(i)[j][k] = old_weight - (LEARNING_RATE)/((double)MINI_BATCH_SIZE) * weight_gradient_matrices.get(i)[j][k];
                        }
                    }
                }

//                System.out.println("PRINT REVISED WEIGHTS AFTER MINI-BATCH");
//                nn.printWeightMatrices();

                // Print overall bias gradients
//                for(int i = 0; i < bias_gradients.size(); ++i){
//                    System.out.println(Arrays.toString(bias_gradients.get(i)));
//                }

                // Revise biases
                for(int i = 0; i < nn.node_layers.size(); ++i){
                    for(int j = 0; j < nn.node_layers.get(i).length; ++j){
                        double old_bias = nn.node_layers.get(i)[j].getBias();
                        double new_bias = old_bias - (LEARNING_RATE)/((double)MINI_BATCH_SIZE) * bias_gradients.get(i)[j];
                        nn.node_layers.get(i)[j].setBias(new_bias);
                    }
                }

//                System.out.println("PRINT REVISED BIASES AFTER MINI-BATCH");
//                for(Node[] layer : nn.node_layers){
//                    for(Node node : layer){
//                        System.out.println(node.getBias());
//                    }
//                }

            } // End of Epoch

            epoch_count += 1;

//            System.out.println("REVISED WEIGHTS AFTER EPOCH");
//            nn.printWeightMatrices();

//            System.out.println("PRINT REVISED BIASES AFTER EPOCH");
//            for(Node[] layer : nn.node_layers){
//                for(Node node : layer){
//                    System.out.println(node.getBias());
//                }
//            }

//            accuracy = ((float)total_correct_in_epoch / (float)8)*100;
//            System.out.println("Num correct: " + correct);
//            System.out.println("Accuracy: " + accuracy + "%");

            accuracy = ((float)total_correct_in_epoch / (float)this.training_data.size())*100;
            printMNISTEpochStats(total_correct_in_epoch, epoch_count, nn_correct_nums, num_instances);
        }

        System.out.println("... Completed!\n");

        //System.out.println("Final biases: ");
        //nn.printBiases();

        //System.out.println();

        //System.out.println("Final weights: ");
        //nn.printSimpleWeightMatrices();


    }
}
