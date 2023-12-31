# AGRODARAA DASHBOARD

## Overview

The AGRODARAA DASHBOARD is a data visualization project designed for displaying smart farm statistics. This repository contains the code and data required to run the dashboard.

## Project Structure

The repository is organized as follows:

- **app.py**: The main Dash application script for the AGRODARAA DASHBOARD.
- **predictor.py**: A class for fetching data, loading models, and making predictions.
- **requirements.txt**: A list of Python dependencies required to run the project.
- **assets/**: A directory containing CSS files for styling the dashboard.
  - **base.css**: General styling for the entire application.
  - **dashboard.css**: Specific styling for the dashboard components.
- **ml_tool/**: A directory containing machine learning models and scalers used in the project.
  - **model.h5**: Model for dataset "5CV-MG3".
  - **scaler.pkl**: Scaler for dataset "5CV-MG3".

## Prerequisites

Before you begin, ensure you have the following prerequisites:

- Python 3.x
- Install the Python packages listed in the `requirements.txt` file by running:
  ```bash
  pip install -r requirements.txt

## Usage

To run the AGRODARAA DASHBOARD, execute the following command:

  ```bash
  python app.py
