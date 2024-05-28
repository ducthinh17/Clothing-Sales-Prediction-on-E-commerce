package com.example.Test;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.evaluation.Evaluation;
import weka.gui.visualize.PlotData2D;
import weka.gui.visualize.VisualizePanel;

import java.awt.BorderLayout;
import java.io.File;
import java.util.Random;

import javax.swing.JFrame;

public class LinearRegressionModel2 {

    public static void main(String[] args) {
        try {
            // Load the dataset
            String filePath = "./Clothing-Sales-Prediction-on-E-commerce/data/cleaned_data.arff";
            File dataFile = new File(filePath);

            if (!dataFile.exists()) {
                System.err.println("File not found: " + filePath);
                return;
            }

            DataSource source = new DataSource(filePath);
            Instances data = source.getDataSet();

            if (data == null) {
                System.err.println("Failed to load data from file: " + filePath);
                return;
            }

            int priceIndex = data.attribute("price").index();
            data.setClassIndex(priceIndex);

            // Randomize the data
            data.randomize(new Random(1));

            // Split the data into training (80%), testing (20%), and validation (last 10
            // records)
            int validationSize = 10;
            int totalSize = data.numInstances();
            int testSize = (int) (totalSize * 0.2);
            int trainSize = totalSize - testSize - validationSize;

            Instances train = new Instances(data, 0, trainSize);
            Instances test = new Instances(data, trainSize, testSize);
            Instances validation = new Instances(data, totalSize - validationSize, validationSize);

            // Create and configure Linear Regression
            LinearRegression lr = new LinearRegression();
            lr.setOptions(new String[] { "-S", "1" }); // Use a method selection for minimizing the error function

            // Build and evaluate the model
            lr.buildClassifier(train);

            // Evaluate classifier with cross-validation
            Evaluation eval = new Evaluation(train);
            eval.crossValidateModel(lr, test, 10, new Random(1));

            // Print model and evaluation results
            System.out.println(lr);
            System.out.println(eval.toSummaryString("\nResults\n======\n", false));
            System.out.println("R^2: " + (eval.correlationCoefficient() * eval.correlationCoefficient()));
            System.out.println("Mean Absolute Error (MAE): " + eval.meanAbsoluteError());
            System.out.println("Root Mean Squared Error (RMSE): " + eval.rootMeanSquaredError());
            System.out.println("Relative Absolute Error: " + eval.relativeAbsoluteError() + "%");
            System.out.println("Root Relative Squared Error: " + eval.rootRelativeSquaredError() + "%\n");

            // Output residuals for inspection
            System.out.println("Actual Value - Linear Regression Model Predicted - Residual");
            for (int i = 0; i < test.numInstances(); i++) {
                double obs = test.instance(i).classValue();
                double pred = lr.classifyInstance(test.instance(i));
                double residual = obs - pred;
                System.out.println(obs + "                " + pred + "          " + residual);
            }

            // Output predictions for the validation set
            System.out.println("Validation Set Predictions:");
            System.out.println("Actual Value - Predicted Value");
            for (int i = 0; i < validation.numInstances(); i++) {
                double actual = validation.instance(i).classValue();
                double predicted = lr.classifyInstance(validation.instance(i));
                System.out.println(actual + "          " + predicted);
            }

            // Visualize the results
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
