package com.example;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Add;

public class RatingClassification {
    public static void main(String[] args) throws Exception {
        // Load dataset
        DataSource source = new DataSource("C:/Users/Admin/Desktop/DATA MINING/Clothing-Sales-Prediction-on-E-commerce-main/data/cleaned_data.arff");
        Instances dataset = source.getDataSet();
        
        // Assuming the 'rating' attribute is at a specific index, change as needed
        int ratingIndex = dataset.attribute("rating").index();
        
        // Add a new nominal attribute for class labels based on rating
        Add filter = new Add();
        filter.setAttributeIndex("last");
        filter.setNominalLabels("good,normal,bad");
        filter.setAttributeName("class_label");
        filter.setInputFormat(dataset);
        dataset = Filter.useFilter(dataset, filter);
        
        // Assign class labels based on rating
        for (int i = 0; i < dataset.numInstances(); i++) {
            double rating = dataset.instance(i).value(ratingIndex);
            String classLabel;
            if (rating >= 4.0) {
                classLabel = "good";
            } else if (rating >= 2.0) {
                classLabel = "normal";
            } else {
                classLabel = "bad";
            }
            dataset.instance(i).setValue(dataset.numAttributes() - 1, classLabel);
        }
        
        // Set class index to the new class attribute
        dataset.setClassIndex(dataset.numAttributes() - 1);
        
        // Split dataset into training and test sets (70% training, 30% testing)
        int trainSize = (int) Math.round(dataset.numInstances() * 0.7);
        int testSize = dataset.numInstances() - trainSize;
        Instances trainDataset = new Instances(dataset, 0, trainSize);
        Instances testDataset = new Instances(dataset, trainSize, testSize);
        
        // Create and build the Naive Bayes classifier
        NaiveBayes nb = new NaiveBayes();
        nb.buildClassifier(trainDataset);
        
        // Evaluate Naive Bayes model
        Evaluation evalNB = new Evaluation(trainDataset);
        evalNB.evaluateModel(nb, testDataset);
        
        System.out.println("===================");
        System.out.println("Naive Bayes Evaluation Results");
        System.out.println(evalNB.toSummaryString("\nResults\n======\n", false));
        System.out.println("Kappa statistic: " + evalNB.kappa());
        System.out.println("Mean absolute error: " + evalNB.meanAbsoluteError());
        System.out.println("Root mean squared error: " + evalNB.rootMeanSquaredError());
        System.out.println("Relative absolute error: " + evalNB.relativeAbsoluteError() + " %");
        System.out.println("Root relative squared error: " + evalNB.rootRelativeSquaredError() + " %");
        System.out.println("Total Number of Instances: " + evalNB.numInstances());
        System.out.println("Confusion Matrix: ");
        double[][] confusionMatrixNB = evalNB.confusionMatrix();
        for (double[] row : confusionMatrixNB) {
            for (double value : row) {
                System.out.print((int) value + " ");
            }
            System.out.println();
        }
        
        // Create and build the SMO classifier
        SMO smo = new SMO();
        smo.buildClassifier(trainDataset);
        
        // Evaluate SMO model
        Evaluation evalSMO = new Evaluation(trainDataset);
        evalSMO.evaluateModel(smo, testDataset);
        
        System.out.println("===================");
        System.out.println("SMO Evaluation Results");
        System.out.println(evalSMO.toSummaryString("\nResults\n======\n", false));
        System.out.println("Kappa statistic: " + evalSMO.kappa());
        System.out.println("Mean absolute error: " + evalSMO.meanAbsoluteError());
        System.out.println("Root mean squared error: " + evalSMO.rootMeanSquaredError());
        System.out.println("Relative absolute error: " + evalSMO.relativeAbsoluteError() + " %");
        System.out.println("Root relative squared error: " + evalSMO.rootRelativeSquaredError() + " %");
        System.out.println("Total Number of Instances: " + evalSMO.numInstances());
        System.out.println("Confusion Matrix: ");
        double[][] confusionMatrixSMO = evalSMO.confusionMatrix();
        for (double[] row : confusionMatrixSMO) {
            for (double value : row) {
                System.out.print((int) value + " ");
            }
            System.out.println();
        }
        
        // Print the model with the higher accuracy
        double nbAccuracy = evalNB.pctCorrect() / 100;
        double smoAccuracy = evalSMO.pctCorrect() / 100;
        
        if (nbAccuracy > smoAccuracy) {
            System.out.println("Naive Bayes is the better model with accuracy: " + nbAccuracy);
        } else {
            System.out.println("SMO is the better model with accuracy: " + smoAccuracy);
        }
    }
}
