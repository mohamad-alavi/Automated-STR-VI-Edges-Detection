# Automated STR-VI Edges Detection and Analysis
This repository contains a Python script designed for automated detection and analysis of wet and dry edges within the Shortwave Infrared Transformed Reflectance (STR) - Vegetation Index (VI) space, commonly known as OPTRAM. The tool enables detailed characterization of land surface dynamics, facilitating the identification of water bodies and vegetation monitoring. Analyzing the relationship between STR and VI accurately identifies wet and dry boundaries, offering critical insights into surface moisture variations and vegetation conditions. Furthermore, this framework serves as a robust foundation for assessing evapotranspiration (ET) by integrating the STR-VI relationship with developed models and auxiliary datasets. Please refer to our publication for more information on estimating evapotranspiration (ET) using the OPTRAM-ETc model.



## Overview

The core functionality of this tool includes:

1.  Data Preprocessing: Applying a series of spectral index-based filters (NDWI, NDVI, EVI, and STR) to prepare the STR data by masking out undesirable areas (e.g., non-vegetated surfaces or cloud-contaminated pixels).
2.  **Vegetation Index (VI) Range Determination:** Automatically establishing the effective range for the VI based on its 1st and 99th percentile values, ensuring a robust analysis across varying landscapes.
3.  **Point Density Filtering:** Identifying and selecting high-density point clusters within the STR-VI feature space, crucial for accurate edge detection, by using `filter_points_by_density` function.
4.  **Edge Coefficient Calculation:** Employing linear regression to compute the slopes and intercepts of both wet and dry edges. The dry edge is determined from the lower bounds and wet edge from the upper bounds of STR values within moving VI intervals, thus defining the bounds of the feature space.
5.  **Visualization:** Creating a scatter plot of STR vs. VI that includes the calculated wet and dry edge lines, enhancing visual interpretation and understanding of the STR-VI relationship using `density_scatter` function.

## Key Features

*   **Automated Processing:** Streamlined data processing and analysis pipeline.
*   **Flexibility:** Supports various vegetation indices by adjusting the `vi_path` variable.
*   **Robust Filtering:** Uses multi-criteria spectral filtering for high-quality data processing.
*   **Density-Based Edge Detection:** Utilizes point density to refine edge detection.
*   **Comprehensive Visualization:** Produces informative scatter plots for effective analysis.

## Repository Structure
content_copy
download
Use code with caution.
Markdown


** Note: The files within the “data/” directory are provided as examples. You should replace them with your actual raster files. The Harmonized Sentinel-2 MSI Surface Reflectance data, which serves as the basis for various spectral indices, can be downloaded directly from [Google Earth Engine](https://code.earthengine.google.com/?scriptPath=users/your_username/your_script) by using this [code](https://code.earthengine.google.com/8d60a101dff9a29531c37233e6ceb2bc).


