package com.example.[Step 3] Random Forest & Evaluation;


import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.Evaluation;

import weka.core.Attribute;
import weka.core.Utils;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class WekaRandomForest {
    public static void main(String[] args) throws Exception {
        // Load dataset
        String datasetPath = "./Clothing-Sales-Prediction-on-E-commerce/data/cleaned_data.arff";
        DataSource source = new DataSource(datasetPath);
        Instances data = source.getDataSet();

        data.sort(data.attribute("units_sold"));
        double Q1 = data.instance((int) (data.numInstances() * 0.25)).value(data.attribute("units_sold"));
        double Q3 = data.instance((int) (data.numInstances() * 0.75)).value(data.attribute("units_sold"));

        // Create new nominal attribute with 3 categories
        List<String> labels = new ArrayList<>();
        labels.add("Slow");
        labels.add("Stable");
        labels.add("Hot");
        Attribute attribute = new Attribute("units_sold_categories", labels);

        int unitsSoldIndex = data.attribute("units_sold").index();
        data.insertAttributeAt(attribute, unitsSoldIndex + 1);
        data.setClassIndex(unitsSoldIndex + 1); // Update class index to new attribute

        // Assign values based on quartile data
        for (int i = 0; i < data.numInstances(); i++) {
            double value = data.instance(i).value(unitsSoldIndex);
            String category = (value <= Q1) ? "Slow" : (value <= Q3) ? "Stable" : "Hot";
            data.instance(i).setValue(unitsSoldIndex + 1, category);
        }

        data.deleteAttributeAt(unitsSoldIndex); // Remove the original 'units_sold' attribute

        // Split data into training and testing
        data.randomize(new Random(1));
        Instances train = new Instances(data, 0, (int) (data.numInstances() * 0.8));
        Instances test = new Instances(data, (int) (data.numInstances() * 0.8),
                (data.numInstances() - (int) (data.numInstances() * 0.8) - 10));
        Instances validation = new Instances(data, data.numInstances() - 10, 10); // Last 10 records for validation

        // Configure and build the Random Forest
        RandomForest forest = new RandomForest();
        String[] options = new String[2];
        options[0] = "-I"; // number of trees
        options[1] = "100"; // trees count
        forest.setOptions(options);
        forest.buildClassifier(train);

        // Evaluate classifier with cross-validation on the test set
        Evaluation eval = new Evaluation(train);
        eval.evaluateModel(forest, test);

        // Print the evaluation results and the confusion matrix
        System.out.println(eval.toSummaryString("\nResults\n======\n", false));
        System.out.println("Confusion Matrix:");
        double[][] confusionMatrix = eval.confusionMatrix();
        for (double[] row : confusionMatrix) {
            for (double element : row) {
                System.out.print(element + " ");
            }
            System.out.println();
        }

        // Print predictions for the validation set
        System.out.println("Validation Set Predictions:");
        for (int i = 0; i < validation.numInstances(); i++) {
            double pred = forest.classifyInstance(validation.instance(i));
            String predicted = validation.classAttribute().value((int) pred);
            String actual = validation.classAttribute().value((int) validation.instance(i).classValue());
            System.out.println("Actual: " + actual + ", Predicted: " + predicted);
        }
    }
}
