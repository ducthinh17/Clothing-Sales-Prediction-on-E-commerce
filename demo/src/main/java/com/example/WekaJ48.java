package com.example;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.trees.J48;
import weka.classifiers.Evaluation;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.core.Attribute;
import weka.core.Instance;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

import guru.nidi.graphviz.engine.Format;
import guru.nidi.graphviz.engine.Graphviz;
import guru.nidi.graphviz.model.MutableGraph;
import guru.nidi.graphviz.parse.Parser;

public class WekaJ48 {
    public static void main(String[] args) throws Exception {
        try {
            // Load CSV
            CSVLoader loader = new CSVLoader();
            loader.setSource(new File(".\\Clothing-Sales-Prediction-on-E-commerce\\data\\cleaned_data.csv")); // Update this to the path of your CSV file
            Instances data = loader.getDataSet();

            // Save ARFF
            ArffSaver saver = new ArffSaver();
            saver.setInstances(data);
            saver.setFile(new File(".\\Clothing-Sales-Prediction-on-E-commerce\\data\\cleaned_data.arff")); // Update this to the desired output path
            saver.writeBatch();

            System.out.println("Conversion successful!");
        } catch (IOException e) {
            e.printStackTrace();
        }

        // Load dataset
        String datasetPath = ".\\Clothing-Sales-Prediction-on-E-commerce\\data\\cleaned_data.arff"; 
        DataSource source = new DataSource(datasetPath);
        Instances data = source.getDataSet();

        int unitsSoldIndex = data.attribute("units_sold").index();
        data.setClassIndex(unitsSoldIndex);

        // Create new nominal attribute with specified categories
        List<String> labels = new ArrayList<>();
        labels.add("Low");
        labels.add("Medium");
        labels.add("High");
        Attribute attribute = new Attribute("units_sold_categories", labels);

        // Replace the old numeric attribute with the new nominal attribute
        data.insertAttributeAt(attribute, unitsSoldIndex + 1);
        data.setClassIndex(unitsSoldIndex + 1);  // Update class index to new attribute

        // Assign values based on original 'units_sold' data
        for (int i = 0; i < data.numInstances(); i++) {
            double value = data.instance(i).value(unitsSoldIndex);
            if (value <= 100) {
                data.instance(i).setValue(unitsSoldIndex + 1, "Low");
            } else if (value <= 500) {
                data.instance(i).setValue(unitsSoldIndex + 1, "Medium");
            } else {
                data.instance(i).setValue(unitsSoldIndex + 1, "High");
            }
        }

        // Remove the original numeric 'units_sold' attribute
        data.deleteAttributeAt(unitsSoldIndex);

        // Random selection of 70% of data as the training set and the rest as testing set
        data.randomize(new java.util.Random(1));
        int trainSize = (int) (data.numInstances() * 0.7);
        int testSize = data.numInstances() - trainSize;
        
        Instances train = new Instances(data, 0, trainSize);
        Instances test = new Instances(data, trainSize, testSize);

        int numClasses = train.numClasses();
        
        //Print out class values in the training set
        for(int i=0; i<numClasses; i++) {
            //Get class string value
            String classValue = train.classAttribute().value(i);
            System.out.println("Class value "+ i +" is "+ classValue);
        }

        // Configure and build the J48 decision tree
        J48 tree = new J48();
        tree.setUnpruned(false);
        tree.setConfidenceFactor(0.25f);
        tree.setMinNumObj(2);
        tree.setBinarySplits(false);
        tree.buildClassifier(train);

        // Export to DOT
        String graph = tree.graph();
        FileWriter fileWriter = new FileWriter("tree.dot");
        fileWriter.write(graph);
        fileWriter.close();

        try {
            // Load the DOT file
            File dotFile = new File(".\\Clothing-Sales-Prediction-on-E-commerce\\tree.dot");

            // Parse the DOT file into a MutableGraph
            MutableGraph g = new Parser().read(dotFile);

            // Render the graph to an image file (e.g., PNG)
            Graphviz.fromGraph(g).width(800).render(Format.PNG).toFile(new File("tree.png"));
            
            System.out.println("Tree visualization saved as tree.png");
        } catch (IOException e) {
            e.printStackTrace();
        }

        // Evaluate classifier with cross-validation
        Evaluation eval = new Evaluation(train);
        eval.crossValidateModel(tree, test, 10, new Random(1));

        // Print the classifier
        System.out.println(tree);

        //Loop through test set and make prediction
        System.out.println("==============");
        System.out.println("Actual Value, C4.5 Decision Tree Predicted");
        for(int i=0; i<test.numInstances(); i++) {
            //Get class double value for current instance
            double actualClass = test.instance(i).classValue();
            //Get class string value using the class index  using the class's int value
            String actual = test.classAttribute().value((int) actualClass);
            //Get Instancce object of current instance
            Instance newInst = test.instance(i);
            //Call classifyInstance, which returns a double value for the class
            double predTree = tree.classifyInstance(newInst);
            //Use this value to get string value of the predicted class
            String predString = test.classAttribute().value((int) predTree);
            System.out.println(actual + ", " + predString);
        }

        // Print the evaluation results
        System.out.println("Detailed Classification Results:");
        System.out.println(eval.toClassDetailsString());
        System.out.println("Summary of Accuracy:");
        System.out.println(eval.toSummaryString());
        System.out.println("Confusion Matrix:");
        System.out.println(eval.toMatrixString());
    }
}

