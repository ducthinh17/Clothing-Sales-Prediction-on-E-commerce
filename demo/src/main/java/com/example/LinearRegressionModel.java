package com.example;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.evaluation.Evaluation;
import weka.gui.visualize.PlotData2D;
import weka.gui.visualize.VisualizePanel;

import java.awt.BorderLayout;
import java.util.Random;

import javax.swing.JFrame;

public class LinearRegressionModel {

    public static void main(String[] args) throws Exception {
        try {
            // Load the dataset
            String filePath = ".\\Clothing-Sales-Prediction-on-E-commerce\\data\\cleaned_data.arff"; 
            DataSource source = new DataSource(filePath);
            Instances data = source.getDataSet();
            
            int priceIndex = data.attribute("price").index();
            data.setClassIndex(priceIndex);

            // Create and configure Linear Regression
            LinearRegression lr = new LinearRegression();

            lr.setOptions(new String[]{"-S", "1"}); // Use a method selection for minimizing the error function

            // Randomize and split the data
            data.randomize(new java.util.Random(1));
            int trainSize = (int) (data.numInstances() * 0.7);
            int testSize = data.numInstances() - trainSize;

            Instances train = new Instances(data, 0, trainSize);
            Instances test = new Instances(data, trainSize, testSize);

            // Build and evaluate the model
            lr.buildClassifier(train);

            // Evaluate classifier with cross-validation
            Evaluation eval = new Evaluation(train);
            eval.crossValidateModel(lr, test, 10, new Random(1));

            // Print model and evaluation results
            System.out.println(lr);
            // Print the evaluation results
            System.out.println(eval.toSummaryString("\nResults\n======\n", false) + "\nR^2: " + eval.correlationCoefficient() * eval.correlationCoefficient() + "\n");

            
            System.out.println("Actual Value - Linear Regression Model Predicted - Residual");
            // Output residuals for inspection
            for (int i = 0; i < test.numInstances(); i++) {
                double obs = test.instance(i).classValue();
                double pred = lr.classifyInstance(test.instance(i));
                double residual = obs - pred;
                System.out.println(obs + "                " + pred + "          " + residual);
            }

            // Visualizing the results
            visualize(lr, train, test);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static void visualize(LinearRegression lr, Instances train, Instances test) throws Exception {
        // Prepare visualization
        PlotData2D plotData = new PlotData2D(test);
        plotData.setPlotName(test.relationName());
        plotData.addInstanceNumberAttribute();

        // Setup visualization
        VisualizePanel vp = new VisualizePanel();
        vp.setName("Visualization of Regression Results");
        vp.addPlot(plotData);

        // Frame for visualization
        JFrame jf = new JFrame("Weka Classifier Visualize: Linear Regression");
        jf.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        jf.setSize(800, 600);
        jf.getContentPane().setLayout(new BorderLayout());
        jf.getContentPane().add(vp, BorderLayout.CENTER);
        jf.setVisible(true);
    }
}

