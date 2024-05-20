package com.example;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Add;
import java.util.Random;

public class RatingClassification2 {
    public static void main(String[] args) throws Exception {
        // Load dataset
        DataSource source = new DataSource("./data/cleaned_data.arff");
        Instances dataset = source.getDataSet();

        // Assuming the 'rating' attribute is at a specific index, change as needed
        int ratingIndex = dataset.attribute("rating").index();

        // Randomize the dataset
        Random rand = new Random(1); // Seed for reproducibility
        dataset.randomize(rand);

        // Separate the last 10 instances for validation before splitting into train and
        // test
        Instances validationSet = new Instances(dataset, dataset.numInstances() - 10, 10);
        dataset = new Instances(dataset, 0, dataset.numInstances() - 10); // Remove validation set from dataset

        // Split dataset into training and test sets (80% training, 20% testing)
        int trainSize = (int) Math.round(dataset.numInstances() * 0.8);
        int testSize = dataset.numInstances() - trainSize;
        Instances trainDataset = new Instances(dataset, 0, trainSize);
        Instances testDataset = new Instances(dataset, trainSize, testSize);

        // Add a new nominal attribute for class labels based on rating
        Add filter = new Add();
        filter.setAttributeIndex("last");
        filter.setNominalLabels("good,normal,bad");
        filter.setAttributeName("class_label");
        filter.setInputFormat(trainDataset); // Ensure the filter is set on the format of the dataset
        trainDataset = Filter.useFilter(trainDataset, filter);
        testDataset = Filter.useFilter(testDataset, filter);
        validationSet = Filter.useFilter(validationSet, filter);

        // Assign class labels based on rating to train, test, and validation datasets
        assignClassLabels(trainDataset, ratingIndex);
        assignClassLabels(testDataset, ratingIndex);
        assignClassLabels(validationSet, ratingIndex);

        // Create and train a Random Forest classifier
        RandomForest rf = new RandomForest();
        rf.buildClassifier(trainDataset);

        // Evaluate Random Forest model on the test dataset
        Evaluation evalRF = new Evaluation(trainDataset);
        evalRF.evaluateModel(rf, testDataset);

        // Print the evaluation results and the confusion matrix
        System.out.println("Random Forest Test Set Evaluation:");
        System.out.println("Correctly Classified Instances: " + (int) evalRF.correct() + "  "
                + String.format("%.4f %%", evalRF.pctCorrect()));
        System.out.println("Incorrectly Classified Instances: " + (int) evalRF.incorrect() + "  "
                + String.format("%.4f %%", evalRF.pctIncorrect()));
        System.out.println("Kappa statistic: " + evalRF.kappa());
        System.out.println("Mean absolute error: " + evalRF.meanAbsoluteError());
        System.out.println("Root mean squared error: " + evalRF.rootMeanSquaredError());
        System.out.println("Relative absolute error: " + evalRF.relativeAbsoluteError() + " %");
        System.out.println("Root relative squared error: " + evalRF.rootRelativeSquaredError() + " %");
        System.out.println("Total Number of Instances: " + evalRF.numInstances());
        System.out.println("Confusion Matrix:");
        double[][] confusionMatrix = evalRF.confusionMatrix();
        for (double[] row : confusionMatrix) {
            for (double value : row) {
                System.out.print((int) value + " ");
            }
            System.out.println();
        }

        // Output predictions for the validation set
        System.out.println("Validation Set Predictions:");
        outputPredictions(rf, validationSet);
    }

    private static void assignClassLabels(Instances data, int ratingIndex) {
        for (int i = 0; i < data.numInstances(); i++) {
            double rating = data.instance(i).value(ratingIndex);
            String classLabel;
            if (rating >= 4.0) {
                classLabel = "good";
            } else if (rating >= 3.0) {
                classLabel = "normal";
            } else {
                classLabel = "bad";
            }
            data.instance(i).setValue(data.numAttributes() - 1, classLabel);
        }
        data.setClassIndex(data.numAttributes() - 1);
    }

    private static void outputPredictions(RandomForest model, Instances data) throws Exception {
        for (int i = 0; i < data.numInstances(); i++) {
            double actual = data.instance(i).classValue();
            double predicted = model.classifyInstance(data.instance(i));
            System.out.println("Actual: " + data.classAttribute().value((int) actual) +
                    ", Predicted: " + data.classAttribute().value((int) predicted));
        }
    }
}
