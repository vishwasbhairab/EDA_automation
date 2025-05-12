# EDA Automation App

![Data Analysis](https://img.shields.io/badge/Data-Analysis-blue)
![Feature Engineering](https://img.shields.io/badge/Feature-Engineering-green)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)

A comprehensive data analysis and feature engineering tool built with Streamlit that automates common exploratory data analysis (EDA) tasks and machine learning preprocessing steps.

## 📊 Features

### Exploratory Data Analysis
- **Basic Statistics:** Comprehensive statistical summaries for both numeric and categorical columns
- **Missing Value Analysis:** Detailed breakdown of missing data across the dataset
- **Correlation Analysis:** Correlation matrix with insights on top positive and negative correlations
- **Distribution Analysis:** Distribution statistics including mean, median, standard deviation, skewness, and kurtosis

### Feature Engineering
- **Missing Value Handling:** Multiple imputation strategies (mean, median, most frequent)
- **Feature Scaling:** Various scaling methods (standard, min-max, robust)
- **Categorical Encoding:** One-hot encoding and label encoding options
- **Feature Interaction:** Automatic generation of polynomial feature interactions
- **Feature Importance:** Estimation of feature importance using mutual information

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- pip

### Installation

1. Clone the repository:
```bash
git clone https://github.com/vishwasbhairab/eda-automation-app.git
cd eda-automation-app
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run EDA_automation_app.py
```

### Dependencies
- streamlit
- pandas
- numpy
- plotly
- seaborn
- matplotlib
- scipy
- scikit-learn

## 💻 Usage

1. Launch the application
2. Upload a CSV file
3. Select an analysis type from the sidebar
4. View the results in the main content area

## 🧪 Example

```python
# Start the EDA Automation App
streamlit run EDA_automation_app.py
```

## 🛠️ Class Structure

The application is built around the `ComprehensiveDataAnalyzer` class, which provides methods for:

- Data exploration
- Missing value analysis
- Feature transformation
- Feature interaction
- Feature importance analysis

## 📝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- This tool integrates common data science libraries like scikit-learn, pandas, and streamlit
- Inspired by the need to automate repetitive EDA tasks in data science workflows
