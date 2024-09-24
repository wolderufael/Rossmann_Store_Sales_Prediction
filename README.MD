# Rossmann Store Sales Prediction

## Table of Contents
- [Project Description](#project-description)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Feature Engineering](#feature-engineering)
- [Modeling Process](#modeling-process)
  - [Data Preprocessing](#data-preprocessing)
  - [Feature Selection](#feature-selection)
  - [Model Development](#model-development)
  - [Model Evaluation](#model-evaluation)
- [Deployment](#deployment)
- [Results](#results)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

---

## Project Description
This project aims to predict daily sales for Rossmann stores across different regions using a variety of features including promotional data, competition distance, store assortment types, and holiday information. The project leverages time-series analysis, machine learning algorithms, and data preprocessing techniques to develop an accurate predictive model. The end goal is to provide Rossmann with actionable insights that will enable them to optimize their promotions and operations based on predicted sales.

The project is based on the Kaggle competition dataset: [Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales).

---

## Project Structure
```plaintext
Rossmann_Store_Sales_Prediction/
│
├── data/                  # Contains the dataset (training and testing data)
│   ├── train.csv
│   ├── test.csv
|   ├── store.csv
│
├── notebooks/             # Jupyter notebooks for data exploration and model development
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│
├── script/                # Python scripts for feature engineering and modeling
│   ├── EDA.py             # Exploratory Data Analysis and Feature Engineering script
│   ├── modeling.py        # Model building and evaluation script
│
├── requirements.txt       # List of required Python packages
├── README.md              # Project's README file
└── LICENSE                # License for the project

```
## Getting Started
### Prerequisites
Before you begin, ensure you have the following installed on your machine:

* Python 3.8+
* Jupyter Notebook (optional, but recommended for exploration)
* Git
### Installation
1. Clone the repository to your local machine:
```
git clone https://github.com/wolderufael/Rossmann_Store_Sales_Prediction.git

```
2. Navigate to the project directory:
```
cd Rossmann_Store_Sales_Prediction

```
3. Create a virtual environment (optional but recommended):
```
python -m venv venv
source venv/bin/activate   # On Windows use `venv\Scripts\activate`

```
4. Install the required dependencies:
```
pip install -r requirements.txt

```
## Usage
To run the project, you can either use the provided notebooks or the Python scripts.

1. **Data Exploration:** Start by exploring the data using the 01_data_exploration.ipynb notebook or by running the EDA.py script for Exploratory Data Analysis (EDA).
```
python script/EDA.py
```
2. **Feature Engineering:** You can create new features using 02_feature_engineering.ipynb notebook or run the EDA.py script, which includes feature engineering.
3. **Modeling:** You can build and evaluate different models using 03_modeling.ipynb or the modeling.py script.

### Feature Engineering
Several features were engineered to improve the model's ability to predict sales accurately:

* **Time-based features:** Day of the week, month, and proximity to holidays.
* **Promo-based features:** Promo2 activation, duration of promotion.
* **Competition features:** Distance to nearest competitor, duration of competitor's presence.
* **Seasonality:** Indicators for holiday seasons (Christmas, Easter, etc.).
The EDA.py script contains all feature engineering processes, including filling missing values and encoding categorical variables.

## Contributing
Contributions are welcome! If you'd like to improve the project, please fork the repository and submit a pull request.

To contribute:

1. Fork the project.
2. Create a new branch.
3. Make your changes.
4. Submit a pull request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.