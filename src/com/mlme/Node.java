/* Author: Daniel Rogers
   Date: 10/16/2019
*/

package com.mlme;

import java.util.Random;


public class Node {
    //private double bias = Math.random(); // random number from 0 to 1
    Random rng = new Random();
    private double bias = rng.nextDouble() * 2 - 1; // get a random # from -1 to 1
    private double activation = 0;

    // Getters and Setters
    public double getActivation() {
        return activation;
    }
    public void setActivation(double activation) {
        this.activation = activation;
    }
    public double getBias() {
        return bias;
    }
    public void setBias(double bias) {
        this.bias = bias;
    }
}
