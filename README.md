# Tpot-analysis
A reference to create an interactive insurance charges prediction dashboard using machine learning with TPOT and Google Data Studio. The workflow encompasses data preprocessing, model training, evaluation, and the creation of a dynamic and informative dashboard.
Please find the Google Colaboratory Notebook in the link below if you face "Unable to render code block" error for the .ipynb file.

Project Link: https://nbviewer.org/github/daluri98/Tpot-analysis/blob/main/predict_charges.ipynb

Dashboard: https://lookerstudio.google.com/reporting/f39616b8-0a41-4ea0-a8bc-2af77abe2bcb/page/p_ddxvtwkjad


## Table of Contents

1. [Project Introduction](#project-introduction)
2. [Prerequisites](#prerequisites)
3. [PART-1 Machine Learning](#part-1-machine-learning)
      - [Data Preprocessing](#data-preprocessing)
      - [Machine Learning and Predictions](#machine-learning-and-predictions)
4. [PART-2 Trends and Insights](#part-2-trends-and-insights)
      - [Visualizing Trends and Insights Using Google Data Studio](#visualizing-trends-and-insights-using-google-data-studio)
5. [Conclusion](#conclusion)
6. [References](#references)

## Project Introduction

This project explores the potential of machine learning and automated data analysis. It leverages the [Tree-based Pipeline Optimization Tool (TPOT)](http://epistasislab.github.io/tpot/) for automating machine learning pipeline optimization and utilizes Google Data Studio to create interactive visualizations. By combining these technologies, the project aims to streamline the process of deriving insights from data and making data-driven decisions.

TPOT is an open-source Python-based (AutoML)](https://www.automl.org/automl/) tool developed by Dr. Randal Olson in 2015. It employs genetic programming to optimize machine learning pipelines. [Genetic programming](https://medium.datadriveninvestor.com/an-introduction-to-genetic-programming-94ad22adbf35
) is an iterative process that selects the best "offspring" based on a fitness function, enabling efficient model development. TPOT integrates with [scikit-learn](https://scikit-learn.org/stable/) components for various tasks and automates the search for optimal machine-learning pipelines.

[Google Data Studio](https://developers.google.com/looker-studio) is an accessible and user-friendly solution to create customizable and interactive dashboards and reports. It offers seamless integration with various data sources and supports collaborative work, making it ideal for visualizing trends and insights from our data.

## Prerequisites
The model training and prediction is run on Google Colaboratoy. It can be accessed [here](https://colab.research.google.com/drive/1J9xTeKDLSAP6aCd1s0-29_M8MYTQX620?usp=sharing) 
Before you can use this code in Google Colab, make sure you have the following prerequisites in place:
1. **Google Colab Account:** Ensure access to Google Colab, a cloud-based Python development environment that can run Jupyter notebooks directly in a web browser.
2. **Google Drive Integration:** This code utilizes Google Drive to access the dataset file and save the best model. Ensure access to a Google Drive account and are logged into Google Colab using the same account for seamless file access.
3. **Python Libraries:** While Google Colab has many shared Python libraries, one might need to install additional libraries if unavailable. Essential libraries used in this code include pandas, scikit-learn, tpot, matplotlib, and seaborn. They are installed with pip using the following commands within the Colab notebook:
```bash
!pip install pandas scikit-learn tpot matplotlib seaborn
```
4. **Dataset:** The dataset used in the project is taken from [Kaggle](https://www.kaggle.com/datasets/mirichoi0218/insurance). The dataset contains 1,338 records. Ensure you have the dataset file (insurance.csv) accessible in your Google Drive. Use the appropriate file path.
5. **Input Data (Optional):** If you intend to make predictions on new data, you can create a DataFrame with the input data. Towards the end of the code, there is an example of creating such a DataFrame named _data_. You can customize this DataFrame with your own input data to predict insurance charges for specific cases.
   
## PART-1 Machine Learning

### Data Preprocessing
Before using the data for machine learning, it's crucial to prepare and preprocess it appropriately. This section outline the steps taken to prepare the dataset for predictive modeling.
**Handling Missing Values**
Check for missing values in the dataset and remove if any. The following code removes any missing values:

```python
missing_values = df.isnull().sum()
```
**Encoding Categorical Variables**
Machine learning models typically require numerical input. Therefore, we encode categorical variables such as 'sex' and 'smoker' using label encoding:
```python
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
df['sex'] = label_encoder.fit_transform(df['sex'])
df['smoker'] = label_encoder.fit_transform(df['smoker'])
```
**One-Hot Encoding**
For categorical data with no ordinal relationship (i.e., the categories have no inherent order or ranking), one-hot encoding is preferred. For the column region, "southwest," "northeast," etc., are region labels with no natural numerical order. If numerical values like 1, 2, 3, and 4 are assigned to these regions, it might imply an ordinal relationship that doesn't exist.
To handle the 'region' column, we use one-hot encoding to convert categorical values into binary columns:
```python
df = pd.get_dummies(df, columns=['region'])
```
These preprocessing steps ensure that the dataset is ready for further analysis and modeling.

### Machine Learning and Predictions
The next steps involve training a machine learning model to make predictions on insurance charges. Here's an overview of the process:

**Model Training with TPOT**
We employ the TPOT to automatically search for and optimize machine learning pipelines. Here's how we configure and train the TPOTRegressor:

```python
from tpot import TPOTRegressor

# Configure TPOTRegressor with parameters
tpot = TPOTRegressor(
    generations=5,              # Number of generations to run
    population_size=20,         # Population size in each generation
    verbosity=2,                # Set verbosity level to see progress
    random_state=42,            # Set random seed for reproducibility
    scoring='neg_mean_squared_error',  # Specify the evaluation metric
    config_dict='TPOT sparse',  # Use a configuration with sparse features
    n_jobs=-1                   # Utilize all available CPU cores
)

# Fit the TPOTRegressor with training data
tpot.fit(X_train, y_train)
```
- A smaller number of generations (generations=5) is chosen to limit the computational time and resources required, especially for demonstration or initial experimentation. Increasing the number of generations can potentially lead to better models but also increases the computational cost.
- A population size of 20 is a reasonable choice for a moderately sized dataset, balancing model quality and computational efficiency.
- Setting n_jobs to -1 instructs TPOT to utilize all available CPU cores, enabling parallelization and faster optimization.

**Model Evaluation**

TPOT selected a _**RandomForestRegressor**_ as the best model after exploring various machine learning pipelines and models. It imples that the random forest regressor yielded the lowest mean squared error (MSE) on the training data. Random forests are known for their robustness and ability to capture complex relationships in data, making them a good choice for regression tasks.After extracting the best pipeline model, the model is evaluated using common regression metrics, including Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared (R2).  However, the accuracy metrics may not have yielded the desired results as the parameters for the _TPOTRegressor_ are set for an experimentation. The accuracy for the model can be improved by further experiementing the parameters or throught methods like Feature Engineering, Hyperparameter Tuning,Ensemble Methods, etc.

## Identifying Factors Affecting Insurance Charges
This section explores the factors that have the most significant influence on insurance charges. Understanding the importance of various features can provide valuable insights for data analysis and decision-making.

```python
# Access the Random Forest regressor from the best model
random_forest = best_model.named_steps['randomforestregressor']

# Get feature importances
feature_importances = random_forest.feature_importances_
```
The code above extracts feature importances from the Random Forest model, which is part of the best model selected by TPOT. The key factors that have the most substantial influence on insurance charges are identified as:

1. **Smoker Status:** Smoking status is the most influential factor affecting insurance charges. Smokers tend to have significantly higher insurance charges compared to non-smokers.

2. **BMI (Body Mass Index):** BMI is another crucial determinant of insurance charges. Higher BMI values are associated with increased insurance costs.

3. **Age:** Age also plays a significant role in insurance charges.

While these three factors—smoker status, BMI, and age—are the primary drivers of insurance charges, it's essential to note that other factors, such as children, region or sex, have a negligible impact in comparison.
Understanding these key factors can guide data analysis and decision-making, helping stakeholders make informed choices regarding insurance pricing and risk assessment.

## PART-2 Trends and Insights

### Visualizing Trends and Insights Using Google Data Studio

We've taken the model trained by TPOT and utilized it to make predictions on the original dataset. This step enables us to analyze the predicted charges and explore trends and insights in Google Data Studio. A new column called predicted_charges is appended to the original DataFrame (original_df). This column contains the predicted insurance charges generated by our machine learning model. The predictions are based on the input features provided in the dataset. To enhance the interpretability of the region data, a new column "region_name" is created. This column maps the numerical region codes (e.g., 1 for northeast, 0 for others) to their corresponding region names (e.g., 'northeast', 'northwest', etc.). This mapping makes it easier to understand and analyze the geographical distribution of insurance charges. This data is saved as a CSV file that can be imported into Google Data Studio to create interactive visualizations, reports, and dashboards,

These data transformation steps ensure that the dataset is enriched with predictive insights while maintaining clarity and comprehensibility for subsequent analysis and reporting.

## Interactive Dashboard for Data Exploration

This project has an simple yet interactive dashboard that allows users to explore and draw insights from the data seamlessly. The dashboard incorporates a variety of charts, graphs, and interactive controls to enhance the data exploration experience.

### Key Features of the Interactive Dashboard:

- **Visualization:** The dashboard includes visually appealing charts and graphs that provide a clear representation of data trends and patterns.
- **Interactivity:** Users can interact with the dashboard by applying filters, selecting data points, and adjusting parameters to focus on specific aspects of the dataset.
- **Insights:** The interactive controls allow users to uncover insights, identify correlations, and answer questions about the dataset.

### Getting Started with the Dashboard:

To begin exploring the dataset and drawing insights, follow these simple steps:

1. [**Access the Dashboard**](https://lookerstudio.google.com/reporting/f39616b8-0a41-4ea0-a8bc-2af77abe2bcb/page/p_ddxvtwkjad)
2. **Navigate:** Familiarize yourself with the dashboard's layout and available controls. 
3. **Interact:** Experiment with sliders, selection options, and parameters to interact with the data.
4. **Analyze:** Use the dashboard to analyze trends, compare data points, and make data-driven decisions.
5. **Save and Share:** If desired, save customized views or share specific insights with others.

The interactive dashboard is a valuable tool for data exploration and decision-making, offering an intuitive and engaging way to interact with the dataset. The dashboard makes the data exploration journey informative and enjoyable for a data analyst, a business stakeholder, or anyone interested in gaining insights.

## Conclusion
This project leverages TPOT to streamline deriving insights from data and making data-driven decisions. Combining automated data analysis with interactive visualizations using Google Data Studio, the project creates a valuable tool for exploring trends, identifying factors affecting insurance charges, and making informed decisions. With a focus on transparency and ease of use, the interactive dashboard allows users to interact with the data seamlessly, making the data exploration informative and enjoyable. 

## References

1. **Genetic Programming:** [Read Introduction to Genetic Programming](https://medium.datadriveninvestor.com/an-introduction-to-genetic-programming-94ad22adbf35)

2. **TPOT (Tree-based Pipeline Optimization Tool) Documentation:** [Official TPOT Documentation](http://epistasislab.github.io/tpot/)

3. **Power BI Microsoft Projects and Examples:** [Power BI Microsoft Projects Examples and Ideas for Practice](https://www.projectpro.io/article/power-bi-microsoft-projects-examples-and-ideas-for-practice/533)

4. **Data Analysis Example During the COVID-19 Pandemic:** [An Analysis of Daily Mortality in France During the COVID-19 Pandemic](https://towardsdatascience.com/an-analysis-of-daily-mortality-in-france-during-the-covid19-pandemic-95286928e80c)

5. **Automated Machine Learning (AutoML) Definition:** [Automated Machine Learning (AutoML) Definition on TechTarget](https://www.techtarget.com/searchenterpriseai/definition/automated-machine-learning-AutoML)





