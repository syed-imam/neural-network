/**
 *  Created by Adil Imam on 11/29/2017
 */

import java.text.*;
import java.util.*;

public class ArtificialNeuralNetwork {

    DecimalFormat df;
    int[] layers;
    Random random = new Random();
    Perceptron bias = new Perceptron();
    double epsilon = 0.0000001;
    double alphaLearningRate = 0.6f;
    double momentum = 0.7f;
    double inputs[][] = { { 1, 1 }, { 1, 0 }, { 0, 1 }, { 0, 0 } };
    double expected[][] = { { 0 }, { 1 }, { 1 }, { 0 } };
    double result[][] = { { -1 }, { -1 }, { -1 }, { -1 } };
    double output[];
    ArrayList<Perceptron> inputLayer = new ArrayList<Perceptron>();
    ArrayList<Perceptron> hiddenLayer = new ArrayList<Perceptron>();
    ArrayList<Perceptron> outputLayer = new ArrayList<Perceptron>();


    public ArtificialNeuralNetwork(int input, int hidden, int output) {
        this.layers = new int[] { input, hidden, output };
        df = new DecimalFormat("#.0#");

        for (int i = 0; i < layers.length; i++) {
            if (i == 0) {
                for (int j = 0; j < layers[i]; j++) {
                    Perceptron perceptron = new Perceptron();
                    inputLayer.add(perceptron);
                }
            } else if (i == 1) {
                for (int j = 0; j < layers[i]; j++) {
                    Perceptron perceptron = new Perceptron();
                    perceptron.addInConnections(inputLayer);
                    perceptron.addBias(bias);
                    hiddenLayer.add(perceptron);
                }
            }

            else if (i == 2) {
                for (int j = 0; j < layers[i]; j++) {
                    Perceptron perceptron = new Perceptron();
                    perceptron.addInConnections(hiddenLayer);
                    perceptron.addBias(bias);
                    outputLayer.add(perceptron);
                }
            } else {
                System.out.println("Error initializing the Network");
            }
        }

        for (Perceptron perceptron : hiddenLayer) {
            ArrayList<Link> links = perceptron.getTotalInConnections();
            for (Link conn : links) {
                double newWeight = getRandom();
                conn.setWeight(newWeight);
            }
        }
        for (Perceptron perceptron : outputLayer) {
            ArrayList<Link> links = perceptron.getTotalInConnections();
            for (Link conn : links) {
                double newWeight = getRandom();
                conn.setWeight(newWeight);
            }
        }
        Perceptron.counter = 0;
        Link.count = 0;
    }

    public static void main(String[] args) {
        ArtificialNeuralNetwork artificialNeuralNetwork = new ArtificialNeuralNetwork(2, 4, 1);
        double minimumError = 0.001;
        int maximumRuns = 20000;
        artificialNeuralNetwork.runANN(maximumRuns, minimumError);
    }

    double getRandom() {
        return 1 * (random.nextDouble() * 2 - 1);
    }

    public void setInput(double inputs[]) {
        for (int i = 0; i < inputLayer.size(); i++) {
            inputLayer.get(i).setOutput(inputs[i]);
        }
    }

    public double[] getOutput() {
        double[] outputs = new double[outputLayer.size()];
        for (int i = 0; i < outputLayer.size(); i++)
            outputs[i] = outputLayer.get(i).getOutput();
        return outputs;
    }

    public void activateFunction() {
        for (Perceptron n : hiddenLayer)
            n.calculateOutput();
        for (Perceptron n : outputLayer)
            n.calculateOutput();
    }



    public void implementBackPropagation(double expectedOutput[]) {
        for (int i = 0; i < expectedOutput.length; i++) {
            double output = expectedOutput[i];
            if (output < 0 || output > 1) {
                if (output < 0)
                    expectedOutput[i] = 0 + epsilon;
                else
                    expectedOutput[i] = 1 - epsilon;
            }
        }

        int i = 0;
        for (Perceptron n : outputLayer) {
            ArrayList<Link> links = n.getTotalInConnections();
            for (Link con : links) {
                double ak = n.getOutput();
                double ai = con.leftPerceptron.getOutput();
                double desiredOutput = expectedOutput[i];
                double partialDerivative = -ak * (1 - ak) * ai * (desiredOutput - ak);
                double deltaWeight = -alphaLearningRate * partialDerivative;
                double newWeight = con.getWeight() + deltaWeight;
                con.setDeltaWeight(deltaWeight);
                con.setWeight(newWeight + momentum * con.getPrevDeltaWeight());
            }
            i++;
        }

        for (Perceptron n : hiddenLayer) {
            ArrayList<Link> links = n.getTotalInConnections();
            for (Link con : links) {
                double aj = n.getOutput();
                double ai = con.leftPerceptron.getOutput();
                double sumKoutputs = 0;
                int j = 0;
                for (Perceptron out_neu : outputLayer) {
                    double wjk = out_neu.getConnection(n.id).getWeight();
                    double desiredOutput = (double) expectedOutput[j];
                    double ak = out_neu.getOutput();
                    j++;
                    sumKoutputs = sumKoutputs
                            + (-(desiredOutput - ak) * ak * (1 - ak) * wjk);
                }

                double partialDerivative = aj * (1 - aj) * ai * sumKoutputs;
                double deltaWeight = -alphaLearningRate * partialDerivative;
                double newWeight = con.getWeight() + deltaWeight;
                con.setDeltaWeight(deltaWeight);
                con.setWeight(newWeight + momentum * con.getPrevDeltaWeight());
            }
        }
    }

    void runANN(int maxSteps, double minError) {
        int i;
        double error = 1;
        for (i = 0; i < maxSteps && error > minError; i++) {
            error = 0;
            for (int p = 0; p < inputs.length; p++) {
                setInput(inputs[p]);
                activateFunction();
                output = getOutput();
                result[p] = output;

                for (int j = 0; j < expected[p].length; j++) {
                    double err = Math.pow(output[j] - expected[p][j], 2);
                    error += err;
                }

                implementBackPropagation(expected[p]);
            }
            System.out.println("Epoch " + i+"\n");
            displayResult();
        }
   }

    void displayResult()
    {
        for (int p = 0; p < inputs.length; p++) {
            System.out.print("Inputs: ");
            for (int x = 0; x < layers[0]; x++) {
                System.out.print(inputs[p][x] + " ");
            }

            System.out.print("Expected: ");
            for (int x = 0; x < layers[2]; x++) {
                System.out.print(expected[p][x] + " ");
            }

            System.out.print("Actual: ");
            for (int x = 0; x < layers[2]; x++) {
                System.out.print(result[p][x] + " ");
            }

            System.out.print("Weights: ");
            for(Perceptron percept: hiddenLayer){
                for(Link l:percept.inconnections){
                     System.out.print(df.format(l.getWeight())+" ");
                }
            }

            System.out.println();
        }
        System.out.println();
    }
}