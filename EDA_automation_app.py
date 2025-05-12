import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Machine Learning and Feature Engineering Imports
from sklearn.preprocessing import (
    StandardScaler, 
    MinMaxScaler, 
    RobustScaler,
    OneHotEncoder, 
    LabelEncoder,
    PolynomialFeatures
)
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,  # Regression metrics
    accuracy_score, precision_score, recall_score, f1_score,  # Classification metrics
    confusion_matrix, classification_report
)

class ComprehensiveDataAnalyzer:
    def __init__(self, dataframe):
        """
        Initialize the Comprehensive Data Analyzer
        
        Args:
            dataframe (pd.DataFrame): Input dataframe to analyze
        """
        self.original_df = dataframe.copy()
        self.df = dataframe.copy()
        self.numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        self.categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns

    # EDA Methods
    def basic_statistics(self):
        """
        Generate comprehensive statistical summary
        
        Returns:
            dict: Descriptive statistics for numeric and categorical columns
        """
        # Numeric columns statistics
        numeric_stats = self.df[self.numeric_cols].describe().T.to_dict()
        
        # Categorical columns statistics
        categorical_stats = {}
        for col in self.categorical_cols:
            categorical_stats[col] = {
                'Unique Values': self.df[col].nunique(),
                'Most Common Value': self.df[col].mode().values[0],
                'Value Counts': dict(self.df[col].value_counts())
            }
        
        return {
            'Numeric Column Statistics': numeric_stats,
            'Categorical Column Statistics': categorical_stats
        }

    def missing_value_analysis(self):
        """
        Comprehensive missing value analysis
        
        Returns:
            dict: Missing value information
        """
        # Missing values
        missing_data = pd.DataFrame({
            'Total Missing': self.df.isnull().sum(),
            'Percent Missing': 100 * self.df.isnull().sum() / len(self.df)
        })
        
        # Filter to only show columns with missing values
        missing_data = missing_data[missing_data['Total Missing'] > 0]
        
        return {
            'Missing Value Summary': missing_data.to_dict(orient='index'),
            'Total Rows': len(self.df),
            'Columns with Missing Values': list(missing_data.index)
        }

    def correlation_analysis(self):
        """
        Perform correlation analysis for numeric columns
        
        Returns:
            dict: Correlation matrix and key insights
        """
        if len(self.numeric_cols) < 2:
            return {"Error": "Not enough numeric columns for correlation analysis"}
        
        # Correlation matrix
        corr_matrix = self.df[self.numeric_cols].corr()
        
        # Find top correlations
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_pairs.append({
                    'Columns': f'{corr_matrix.columns[i]} - {corr_matrix.columns[j]}',
                    'Correlation': corr_matrix.iloc[i, j]
                })
        
        # Sort correlations
        top_positive_corr = sorted(
            [p for p in corr_pairs if p['Correlation'] > 0], 
            key=lambda x: x['Correlation'], 
            reverse=True
        )[:5]
        
        top_negative_corr = sorted(
            [p for p in corr_pairs if p['Correlation'] < 0], 
            key=lambda x: x['Correlation']
        )[:5]
        
        return {
            'Correlation Matrix': corr_matrix.to_dict(),
            'Top Positive Correlations': top_positive_corr,
            'Top Negative Correlations': top_negative_corr
        }

    def distribution_analysis(self):
        """
        Analyze distributions of numeric columns
        
        Returns:
            dict: Distribution statistics and skewness
        """
        distribution_stats = {}
        
        for col in self.numeric_cols:
            # Basic distribution statistics
            distribution_stats[col] = {
                'Mean': self.df[col].mean(),
                'Median': self.df[col].median(),
                'Standard Deviation': self.df[col].std(),
                'Skewness': self.df[col].skew(),
                'Kurtosis': self.df[col].kurtosis()
            }
        
        return distribution_stats

    # Feature Engineering Methods
    def handle_missing_values(self, strategy='median'):
        """
        Handle missing values with various imputation strategies
        
        Args:
            strategy (str): Imputation strategy 
                            ('mean', 'median', 'most_frequent', 'constant')
        
        Returns:
            dict: Missing value handling results
        """
        # Initial missing value check
        missing_before = self.df.isnull().sum()
        
        # Impute numeric columns
        numeric_imputer = SimpleImputer(strategy=strategy)
        self.df[self.numeric_cols] = numeric_imputer.fit_transform(self.df[self.numeric_cols])
        
        # Impute categorical columns
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        self.df[self.categorical_cols] = categorical_imputer.fit_transform(self.df[self.categorical_cols])
        
        # Check missing values after imputation
        missing_after = self.df.isnull().sum()
        
        return {
            'Missing Values Before': dict(missing_before),
            'Missing Values After': dict(missing_after),
            'Imputation Strategy': strategy
        }

    def feature_scaling(self, method='standard'):
        """
        Apply feature scaling to numeric columns
        
        Args:
            method (str): Scaling method 
                          ('standard', 'minmax', 'robust')
        
        Returns:
            dict: Scaling information
        """
        # Apply scaling
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError("Invalid scaling method")
        
        # Scale numeric columns
        scaled_data = scaler.fit_transform(self.df[self.numeric_cols])
        scaled_df = pd.DataFrame(
            scaled_data, 
            columns=[f'{col}_scaled' for col in self.numeric_cols]
        )
        
        # Add scaled columns to dataframe
        self.df = pd.concat([self.df, scaled_df], axis=1)
        
        return {
            'Scaling Method': method,
            'Scaled Columns': list(scaled_df.columns)
        }

    def categorical_encoding(self, encoding_method='onehot'):
        """
        Encode categorical variables
        
        Args:
            encoding_method (str): Encoding method 
                                   ('onehot', 'label')
        
        Returns:
            dict: Encoding information
        """
        # Prepare for encoding
        encoded_columns = []
        
        if encoding_method == 'onehot':
            # One-Hot Encoding
            onehot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            
            for col in self.categorical_cols:
                # Perform one-hot encoding
                encoded = onehot_encoder.fit_transform(self.df[[col]])
                
                # Create column names
                encoded_col_names = [f'{col}_{cat}' for cat in onehot_encoder.categories_[0]]
                
                # Convert to DataFrame
                encoded_df = pd.DataFrame(
                    encoded, 
                    columns=encoded_col_names, 
                    index=self.df.index
                )
                
                # Add to dataframe
                self.df = pd.concat([self.df, encoded_df], axis=1)
                encoded_columns.extend(encoded_col_names)
        
        elif encoding_method == 'label':
            # Label Encoding
            label_encoder = LabelEncoder()
            
            for col in self.categorical_cols:
                # Perform label encoding
                self.df[f'{col}_encoded'] = label_encoder.fit_transform(self.df[col].astype(str))
                encoded_columns.append(f'{col}_encoded')
        
        else:
            raise ValueError("Invalid encoding method")
        
        return {
            'Encoding Method': encoding_method,
            'Encoded Columns': encoded_columns
        }

    def feature_interaction(self):
        """
        Create interaction features for numeric columns
        
        Returns:
            dict: Feature interaction information
        """
        # Polynomial Features
        poly = PolynomialFeatures(degree=2, include_bias=False)
        
        # Create interaction features
        interaction_data = poly.fit_transform(self.df[self.numeric_cols])
        
        # Create column names
        feature_names = poly.get_feature_names_out(self.numeric_cols)
        
        # Create DataFrame with interaction features
        interaction_df = pd.DataFrame(
            interaction_data, 
            columns=feature_names, 
            index=self.df.index
        )
        
        # Add to original dataframe
        self.df = pd.concat([self.df, interaction_df], axis=1)
        
        return {
            'Interaction Features': list(interaction_df.columns)
        }

    def feature_importance(self, target_column=None, problem_type='auto'):
        """
        Estimate feature importance using mutual information
        
        Args:
            target_column (str): Target variable for importance estimation
            problem_type (str): 'auto', 'classification', or 'regression'
        
        Returns:
            dict: Feature importance information
        """
        if target_column is None:
            return {"Error": "No target column specified"}
        
        # Prepare features and target
        X = self.df.drop(columns=[target_column])
        y = self.df[target_column]
        
        # Detect problem type if 'auto'
        if problem_type == 'auto':
            problem_type = 'classification' if y.dtype == 'object' or len(np.unique(y)) < 10 else 'regression'
        
        # Calculate mutual information
        if problem_type == 'classification':
            # Label encode target if categorical
            le = LabelEncoder()
            y = le.fit_transform(y.astype(str))
            
            # Mutual information for classification
            importance = mutual_info_classif(X, y)
        else:
            # Mutual information for regression
            importance = mutual_info_regression(X, y)
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        return {
            'Problem Type': problem_type,
            'Feature Importance': importance_df.to_dict(orient='records')
        }

def main():
    st.set_page_config(page_title='Comprehensive Data Analysis & Feature Engineering', layout='wide')
    
    st.title('🔬 Comprehensive Data Analysis & Feature Engineering Tool')
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)
        
        # Initialize Comprehensive Data Analyzer
        data_analyzer = ComprehensiveDataAnalyzer(df)
        
        # Sidebar for navigation
        analysis_type = st.sidebar.radio(
            "Select Analysis Type", 
            [
                "Basic Statistics",
                "Missing Value Analysis",
                "Correlation Analysis", 
                "Distribution Analysis",
                "Handle Missing Values",
                "Feature Scaling",
                "Categorical Encoding",
                "Feature Interaction",
                "Feature Importance"
            ]
        )
        
        # Main content area
        st.subheader(f"Analysis: {analysis_type}")
        
        if analysis_type == "Basic Statistics":
            stats = data_analyzer.basic_statistics()
            st.json(stats)
        
        elif analysis_type == "Missing Value Analysis":
            missing_analysis = data_analyzer.missing_value_analysis()
            st.json(missing_analysis)
        
        elif analysis_type == "Correlation Analysis":
            correlation_results = data_analyzer.correlation_analysis()
            st.json(correlation_results)
        
        elif analysis_type == "Distribution Analysis":
            distribution_results = data_analyzer.distribution_analysis()
            st.json(distribution_results)
        
        elif analysis_type == "Handle Missing Values":
            # Imputation strategy selection
            imputation_strategy = st.selectbox(
                "Select Imputation Strategy", 
                ['median', 'mean', 'most_frequent']
            )
            missing_handling_results = data_analyzer.handle_missing_values(strategy=imputation_strategy)
            st.json(missing_handling_results)
        
        elif analysis_type == "Feature Scaling":
            # Scaling method selection
            scaling_method = st.selectbox(
                "Select Scaling Method", 
                ['standard', 'minmax', 'robust']
            )
            scaling_results = data_analyzer.feature_scaling(method=scaling_method)
            st.json(scaling_results)
        
        elif analysis_type == "Categorical Encoding":
            # Encoding method selection
            encoding_method = st.selectbox(
                "Select Encoding Method", 
                ['onehot', 'label']
            )
            encoding_results = data_analyzer.categorical_encoding(encoding_method=encoding_method)
            st.json(encoding_results)
        
        elif analysis_type == "Feature Interaction":
            interaction_results = data_analyzer.feature_interaction()
            st.json(interaction_results)
        
        elif analysis_type == "Feature Importance":
            # Target column selection
            target_column = st.selectbox(
                "Select Target Column", 
                df.columns.tolist()
            )
            
            # Problem type selection
            problem_type = st.selectbox(
                "Select Problem Type", 
                ['auto', 'classification', 'regression']
            )
            
            if target_column:
                importance_results = data_analyzer.feature_importance(
                    target_column, 
                    problem_type
                )
                st.json(importance_results)

if __name__ == "__main__":
    main()