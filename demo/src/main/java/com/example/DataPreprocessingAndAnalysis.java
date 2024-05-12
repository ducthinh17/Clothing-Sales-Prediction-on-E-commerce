package com.example;

import com.opencsv.CSVReader;
import com.opencsv.CSVWriter;
import com.opencsv.exceptions.CsvException;
import com.opencsv.exceptions.CsvValidationException;
import org.apache.commons.lang3.StringUtils;
import org.apache.commons.lang3.math.NumberUtils;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

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
            List<String[]> allData = reader.readAll();
            if (allData.size() <= 1) {
                System.out.println("No data available for analysis.");
                return;
            }
            
            List<Double> prices = new ArrayList<>();
            
            // Skip header row
            for (int i = 1; i < allData.size(); i++) {
                String[] row = allData.get(i);

                try {
                    double price = Double.parseDouble(row[2]);
                    prices.add(price);
                } catch (NumberFormatException e) {
                    System.out.println("Invalid price format at row " + i + ": " + row[2]);
                }
            }

            // Calculate and display statistics for prices
            if (!prices.isEmpty()) {
                Collections.sort(prices);
                double median = calculateMedian(prices);
                double mean = prices.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
                double stdDev = calculateStandardDeviation(prices, mean);
                System.out.println("Average Price: $" + mean);
                System.out.println("Median Price: $" + median);
                System.out.println("Standard Deviation of Prices: $" + stdDev);
            } else {
                System.out.println("No valid price data available.");
            }
        }
    }

    private static double calculateMedian(List<Double> prices) {
        int middle = prices.size() / 2;
        if (prices.size() % 2 == 0) {
            return (prices.get(middle - 1) + prices.get(middle)) / 2.0;
        } else {
            return prices.get(middle);
        }
    }

    private static double calculateStandardDeviation(List<Double> prices, double mean) {
        double sum = 0.0;
        for (double price : prices) {
            sum += Math.pow(price - mean, 2);
        }
        return Math.sqrt(sum / prices.size());
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
