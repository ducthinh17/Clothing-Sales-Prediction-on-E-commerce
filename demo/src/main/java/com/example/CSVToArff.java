package com.example;
import java.io.File;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;

public class CSVToArff {
    public static void main(String[] args) throws Exception {
        // Đường dẫn tới tệp CSV
        String csvFilePath = "C:/Users/Admin/Desktop/DATA MINING/Clothing-Sales-Prediction-on-E-commerce-main/data/cleaned_data.csv";
        // Đường dẫn để lưu tệp ARFF
        String arffFilePath = "C:/Users/Admin/Desktop/DATA MINING/Clothing-Sales-Prediction-on-E-commerce-main/data/cleaned_data.arff";

        // Tải dữ liệu từ tệp CSV
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(csvFilePath));
        Instances data = loader.getDataSet();

        // Lưu dữ liệu dưới dạng tệp ARFF
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File(arffFilePath));
        saver.writeBatch();

        System.out.println("Chuyển đổi thành công tệp CSV sang ARFF!");
    }
}
