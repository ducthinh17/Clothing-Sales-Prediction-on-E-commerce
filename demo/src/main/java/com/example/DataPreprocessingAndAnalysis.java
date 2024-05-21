package com.example;

import com.opencsv.CSVReader;
import com.opencsv.CSVWriter;
import com.opencsv.exceptions.CsvException;
import com.opencsv.exceptions.CsvValidationException;
import org.apache.commons.lang3.StringUtils;
import org.apache.commons.lang3.math.NumberUtils;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics; 

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartUtils;
import org.jfree.chart.JFreeChart;
import org.jfree.data.statistics.HistogramDataset;
import org.jfree.data.statistics.HistogramType;

public class DataPreprocessingAndAnalysis {
    public static void main(String[] args) throws FileNotFoundException, IOException, CsvException {
        String inputFilePath = ".\\Data Mining Project\\data\\summer-products-with-rating-and-performance_2020-08.csv";
        String outputFilePath = ".\\Data Mining Project\\data\\cleaned_data.csv";
        String columnToEncode = "product_color";

        // Define columns to drop
        Set<String> columnsToDrop = new HashSet<>(Arrays.asList(
            "title", "title_orig", "currency_buyer", "shipping_option_name", "urgency_text",
            "merchant_title", "merchant_name", "merchant_info_subtitle", "merchant_id",
            "merchant_profile_picture", "product_url", "product_picture", "product_id",
            "tags", "has_urgency_banner", "theme", "crawl_month", "origin_country"
        ));

        try (CSVReader reader = new CSVReader(new FileReader(inputFilePath));
             CSVWriter writer = new CSVWriter(new FileWriter(outputFilePath))) {

            String[] headers = reader.readNext();
            List<Integer> indicesToKeep = new ArrayList<>();
            Map<Integer, List<String>> columnData = new HashMap<>();
            Set<List<String>> uniqueRows = new HashSet<>();

            // Identify indices of columns to keep
            for (int i = 0; i < headers.length; i++) {
                if (!columnsToDrop.contains(headers[i])) {
                    indicesToKeep.add(i);
                }
            }

            // Read and collect data for each relevant column
            String[] row;
            while ((row = reader.readNext()) != null) {
                List<String> filteredRow = new ArrayList<>();
                for (int index : indicesToKeep) {
                    filteredRow.add(row[index]);
                    columnData.computeIfAbsent(index, k -> new ArrayList<>()).add(row[index]);
                }
                uniqueRows.add(filteredRow);  // Add row to set to prevent duplicates
            }

            // Process and prepare data for output
            List<String[]> processedData = new ArrayList<>();
            processedData.add(Arrays.stream(headers).filter(h -> !columnsToDrop.contains(h)).toArray(String[]::new));

            // Calculate replacements for missing values
            Map<Integer, String> replacements = calculateReplacements(columnData);

            Map<String, Integer> encodingMap = encodeSizes();

            String normalizeColor = normalizeColor(columnToEncode);

            for (List<String> uniqueRow : uniqueRows) {
                String[] processedRow = processRow(uniqueRow, indicesToKeep, replacements, encodingMap, normalizeColor, headers);
                processedData.add(processedRow);
            }

            List<String[]> newData = oneHotEncode(processedData, columnToEncode);

            // Write the processed data to a new CSV file
            writer.writeAll(newData);

        } catch (IOException | CsvValidationException e) {
            e.printStackTrace();
        }

        try (CSVReader reader = new CSVReader(new FileReader(outputFilePath))) {
            List<String[]> records = reader.readAll();
            DescriptiveStatistics priceStats = new DescriptiveStatistics();
            double[] unitsSoldValues = new double[records.size() - 1];
            double[] ratingValues = new double[records.size() - 1];
            
            // Identify the indices for 'price', 'units_sold' and 'rating' columns
            String[] headers = records.get(0);
            int priceIndex = -1, unitsSoldIndex = -1, ratingIndex = -1;
            for (int i = 0; i < headers.length; i++) {
                if (headers[i].equals("price")) {
                    priceIndex = i;
                } else if (headers[i].equals("units_sold")) {
                    unitsSoldIndex = i;
                } else if (headers[i].equals("rating")) {
                    ratingIndex = i;
                }    
            }

            for (int i = 1; i < records.size(); i++) {
                String[] row = records.get(i);
                if (priceIndex != -1) {
                    double price = Double.parseDouble(row[priceIndex]);
                    priceStats.addValue(price);
                }
                if (unitsSoldIndex != -1) {
                    double unitsSold = Double.parseDouble(row[unitsSoldIndex]);
                    unitsSoldValues[i - 1] = unitsSold;
                }
                if (ratingIndex != -1) {
                    double rating = Double.parseDouble(row[ratingIndex]);
                    ratingValues[i - 1] = rating;
                }
            }

            System.out.println("Price - Mean: " + priceStats.getMean());
            System.out.println("Price - Median: " + priceStats.getPercentile(50));
            System.out.println("Price - Standard Deviation: " + priceStats.getStandardDeviation());
            System.out.println("Price - Variance: " + priceStats.getVariance());

            // Create a histogram for 'units_sold'
            HistogramDataset us = new HistogramDataset();
            us.setType(HistogramType.RELATIVE_FREQUENCY);
            us.addSeries("Units Sold", unitsSoldValues, 10); // 10 bins

            JFreeChart unitsSoldHistogram = ChartFactory.createHistogram(
                "Units Sold Distribution",
                "Units Sold",
                "Frequency",
                us
            );

            // Create a histogram for 'rating'
            HistogramDataset ratingData = new HistogramDataset();
            ratingData.setType(HistogramType.RELATIVE_FREQUENCY);
            ratingData.addSeries("Rating", ratingValues, 10); // 10 bins

            JFreeChart ratingHistogram = ChartFactory.createHistogram(
                "Rating Distribution",
                "Rating",
                "Frequency",
                ratingData
            );

            // Save the histogram as a PNG file
            ChartUtils.saveChartAsPNG(new java.io.File("units_sold_histogram.png"), unitsSoldHistogram, 500, 300);
            ChartUtils.saveChartAsPNG(new java.io.File("rating_histogram.png"), ratingHistogram, 500, 300);
            
        }
    }

    private static List<String[]> oneHotEncode(List<String[]> data, String columnToEncode) {
        int columnIndex = -1;
        Set<String> uniqueValues = new HashSet<>();
        List<String[]> result = new ArrayList<>();

        // Determine the index of the column to encode and find all unique values
        String[] headers = data.get(0);
        for (int i = 0; i < headers.length; i++) {
            if (headers[i].equals(columnToEncode)) {
                columnIndex = i;
                break;
            }
        }

        if (columnIndex == -1) {
            throw new IllegalArgumentException("Column not found: " + columnToEncode);
        }

        for (int i = 1; i < data.size(); i++) {
            uniqueValues.add(data.get(i)[columnIndex]);
        }

        // Create new headers for one-hot encoded columns
        List<String> newHeaders = new ArrayList<>(Arrays.asList(headers));
        newHeaders.remove(columnIndex);
        uniqueValues.forEach(value -> newHeaders.add(value));
        result.add(newHeaders.toArray(new String[0]));

        // Encode data
        for (int i = 1; i < data.size(); i++) {
            List<String> newRow = new ArrayList<>(Arrays.asList(data.get(i)));
            String value = newRow.remove(columnIndex);
            uniqueValues.forEach(uniqueValue -> {
                newRow.add(value.equals(uniqueValue) ? "1" : "0");
            });
            result.add(newRow.toArray(new String[0]));
        }

        return result;
    }

    private static Map<Integer, String> calculateReplacements(Map<Integer, List<String>> columnData) {
        Map<Integer, String> replacements = new HashMap<>();
        columnData.forEach((index, data) -> {
            if (NumberUtils.isCreatable(data.get(0))) {
                double mean = data.stream().filter(StringUtils::isNotBlank)
                                  .mapToDouble(Double::parseDouble).average().orElse(0);
                replacements.put(index, String.format("%.2f", mean));
            } else {
                String mostCommon = data.stream().filter(StringUtils::isNotBlank)
                                  .collect(Collectors.groupingBy(Function.identity(), Collectors.counting()))
                                  .entrySet().stream().max(Map.Entry.comparingByValue())
                                  .map(Map.Entry::getKey).orElse("Unknown");
                replacements.put(index, mostCommon);
            }
        });
        return replacements;
    }

    private static String[] processRow(List<String> row, List<Integer> indicesToKeep, Map<Integer, String> replacements, Map<String, Integer> encodingMap, String normalizeColor, String[] headers) {
        String[] processedRow = new String[indicesToKeep.size()];
        for (int i = 0; i < indicesToKeep.size(); i++) {
            int index = indicesToKeep.get(i);
            String value = row.get(i);

            // Fill missing values
            if (StringUtils.isBlank(value)) {
                value = replacements.get(index);
            }

            // Normalize specific column
            if (headers[index].equals("product_variation_size_id")) {
                value = String.valueOf(encodingMap.get(normalizeSize(value)));
            }

            // Convert currency if needed
            if (headers[index].equals("price")) {
                value = String.valueOf(convertCurrency(Double.parseDouble(value), "USD", "EUR"));
            }

            if (headers[index].equals("product_color")) {
                value = normalizeColor(value);
            }

            processedRow[i] = value;
        }
        return processedRow;
    }

    private static String normalizeSize(String size) {
        if (size == null) return "Unknown"; // Handle null values
    
        size = size.trim().toUpperCase(); // Normalize case and remove whitespace
    
        Map<String, String> sizeMap = new HashMap<>();
        sizeMap.put("S", "S");
        sizeMap.put("SMALL", "S");
        sizeMap.put("SML", "S");
        sizeMap.put("XS", "XS");
        sizeMap.put("EXTRA SMALL", "XS");
        sizeMap.put("M", "M");
        sizeMap.put("MEDIUM", "M");
        sizeMap.put("MED", "M");
        sizeMap.put("L", "L");
        sizeMap.put("LARGE", "L");
        sizeMap.put("LRG", "L");
        sizeMap.put("XL", "XL");
        sizeMap.put("EXTRA LARGE", "XL");
        sizeMap.put("XXL", "XXL");
        sizeMap.put("2XL", "XXL");
        sizeMap.put("DOUBLE XL", "XXL");
    
        return sizeMap.getOrDefault(size, "Other");
    }

    //Encode normalized sizes
    private static Map<String, Integer> encodeSizes() {
        Map<String, Integer> encodingMap = new HashMap<>();
        encodingMap.put("XS", 1);
        encodingMap.put("S", 2);
        encodingMap.put("M", 3);
        encodingMap.put("L", 4);
        encodingMap.put("XL", 5);
        encodingMap.put("XXL", 6);
        encodingMap.put("Unknown", 7);
        encodingMap.put("Other", 8); // Use for unexpected or new sizes
    
        return encodingMap;
    }

    private static final Set<String> baseColors = initializeBaseColors();

    private static String normalizeColor(String color) {
        // Remove all non-alphanumeric characters (keep only letters and numbers)
        String normalized = color.replaceAll("[^a-zA-Z0-9]", "").toLowerCase();

        if (colorMap.containsKey(normalized)) {
            return colorMap.get(normalized);
        }

        // Check for base color inclusion and map accordingly
        for (String baseColor : baseColors) {
            if (normalized.contains(baseColor)) {
                return baseColor;
            }
        }
        return normalized; // return as is if no base color is found
    }

    private static Set<String> initializeBaseColors() {
        return new HashSet<>(Arrays.asList(
            "blue", "green", "red", "yellow", "orange", "purple", "pink", "white", "black", "grey", "beige", "brown"
        ));
    }

    private static final Map<String, String> colorMap = initializeColorMap();

    private static Map<String, String> initializeColorMap() {
        Map<String, String> map = new HashMap<>();
        // Detailed normalization of color variations
        map.put("navy", "blue");
        map.put("khaki", "beige");
        map.put("lightkhaki", "beige");
        map.put("nude", "beige");
        map.put("tan", "beige");
        map.put("apricot", "orange");
        map.put("cream", "white");
        map.put("ivory", "white");
        map.put("camouflage", "green");
        map.put("army", "green");
        map.put("gold", "yellow");
        map.put("star", "yellow");
        map.put("claret", "red");
        map.put("wine", "red");
        map.put("burgundy", "red");
        map.put("lightgray", "grey");
        map.put("silver", "grey");
        map.put("violet", "purple");
        map.put("rosegold", "pink");
        map.put("rose", "pink");
        map.put("leopardprint", "multicolor");
        map.put("leopard", "multicolor");
        map.put("jasper", "multicolor");
        map.put("rainbow", "multicolor");
        map.put("floral", "multicolor");
        map.put("camel", "brown");
        map.put("coffee", "brown");

        // Add more mappings as needed
        return map;
    }

    private static double convertCurrency(double amount, String fromCurrency, String toCurrency) {
        return amount * 1.1; // Example conversion rate, replace with actual API or conversion logic as needed
    }
}
