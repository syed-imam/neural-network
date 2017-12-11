public class Link {
    double weight = 0;
    double prevDeltaWeight = 0;
    double deltaWeight = 0;
    Perceptron leftPerceptron;
    Perceptron rightPerceptron;
    static int count = 0;
    final public int id;
 
    public Link(Perceptron from, Perceptron to) {
        leftPerceptron = from;
        rightPerceptron = to;
        id = count;
        count++;
    }
 
    public double getWeight() {
        return weight;
    }
 
    public void setWeight(double w) {
        weight = w;
    }
 
    public void setDeltaWeight(double w) {
        prevDeltaWeight = deltaWeight;
        deltaWeight = w;

    }
 
    public double getPrevDeltaWeight() {
        return prevDeltaWeight;
    }
 
    public Perceptron getFromNeuron() {
        return leftPerceptron;
    }
 
    public Perceptron getToNeuron() {
        return rightPerceptron;
    }
}