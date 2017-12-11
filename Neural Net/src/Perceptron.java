import java.util.*;
 
public class Perceptron {
    static int counter = 0;
    final public int id;  // auto increment, starts at 0
    Link biasLink;
    final double bias = -1;
    double output;
     
    ArrayList<Link> inconnections = new ArrayList<Link>();
    HashMap<Integer, Link> connectionLookup = new HashMap<Integer, Link>();
     
    public Perceptron(){
        id = counter;
        counter++;
    }

    public ArrayList<Link> getTotalInConnections(){
        return inconnections;
    }

    public double getOutput() {
        return output;
    }

    public void setOutput(double o){
        output = o;
    }


    public void calculateOutput(){
        double value = 0;
        for(Link con : inconnections){
            Perceptron leftPerceptron = con.getFromNeuron();
            double weight = con.getWeight();
            double a = leftPerceptron.getOutput();
            value = value + (weight*a);
        }
        value = value + (biasLink.getWeight()*bias);
        output = sigmoid(value);
    }

    double sigmoid(double x) {
        return 1.0 / (1.0 +  (Math.exp(-x)));
    }
     
    public void addInConnections(ArrayList<Perceptron> inPerceptrons){
        for(Perceptron n: inPerceptrons){
            Link con = new Link(n,this);
            inconnections.add(con);
            connectionLookup.put(n.id, con);
        }
    }
     
    public Link getConnection(int index){
        return connectionLookup.get(index);
    }
 
    public void addBias(Perceptron p){
        Link con = new Link(p,this);
        biasLink = con;
        inconnections.add(con);
    }
}