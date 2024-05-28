package com.example.[Step 2] Naive Bayes &  Evaluation ;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class WekaNaiveBayes {
    public static void main(String[] args) throws Exception {
      
        //Dataset
        String datasetPath = "./Clothing-Sales-Prediction-on-E-commerce-main/data/cleaned_data.arff";
        DataSource source = new DataSource(datasetPath);
        Instances data = source.getDataSet();

        data.sort(data.attribute("units_sold"));
        double Q1 = data.instance((int) (data.numInstances() * 0.25)).value(data.attribute("units_sold"));
        double Q3 = data.instance((int) (data.numInstances() * 0.75)).value(data.attribute("units_sold"));

        // Create new nominal attribute with specified categories
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

        // Configure and build the Naive Bayes classifier
        NaiveBayes nb = new NaiveBayes();
        nb.buildClassifier(train);

        // Evaluate classifier with cross-validation on the test set
        Evaluation eval = new Evaluation(train);
        eval.evaluateModel(nb, test);

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
            double pred = nb.classifyInstance(validation.instance(i));
            String predicted = validation.classAttribute().value((int) pred);
            String actual = validation.classAttribute().value((int) validation.instance(i).classValue());
            System.out.println("Actual: " + actual + ", Predicted: " + predicted);
        }
    }
}
