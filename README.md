# K-Means Clustering Application  

## Overview  
This project implements a Python-based K-Means clustering application to analyze and uncover patterns in customer review data. The goal is to organize reviews based on their helpfulness scores, providing actionable insights to enhance business strategies.  

## Features  
- **Data Preprocessing**: Handles missing values, standardizes numerical data, and prepares it for clustering.  
- **K-Means Clustering**: Groups data into clusters based on helpfulness scores.  
- **Visualization**: Generates histograms to illustrate the distribution of helpfulness scores across clusters.  
- **Scalability**: The application is designed to handle similar datasets for clustering tasks.  

## Dataset  
- **Source**: AWS Open Data Registry  
- **File**: `train.json`  
- **URL**: [train.json dataset from AWS Open Data Registry](https://s3.amazonaws.com/helpful-sentences-from-reviews/train.json)  
- **Columns**:  
  - `asin`: Product identifier.  
  - `sentence`: Review text.  
  - `helpful`: Numeric scores reflecting review helpfulness.  
  - `main_image_url`: Links to product images.  
  - `product_title`: Titles of the reviewed products.  

## Installation  
1. Clone the repository:  
   ```bash
   git clone https://github.com/romeofx/k-means-clustering.git
   cd k-means-clustering

