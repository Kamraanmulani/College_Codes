
# Demand Forecasting Using Regression Analysis
# Assignment 6: Flip Classroom Activity
# Course: Machine Learning / Data Science
# Date: September 17, 2025

"""
This script implements a comprehensive regression analysis to understand demand patterns
and optimize company revenue using machine learning techniques.
"""

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

def main():
    """Main function to execute the demand forecasting analysis"""

    print("=" * 60)
    print("DEMAND FORECASTING USING REGRESSION ANALYSIS")
    print("=" * 60)
    print("📊 Starting comprehensive analysis...")

    # 1. Data Loading and Preparation
    print("\n1. 📥 LOADING AND PREPARING DATA")
    print("-" * 40)

    try:
        # Load the dataset (download from: https://www.kaggle.com/datasets/yasserh/advertising-sales-dataset)
        # Make sure to place 'Advertising_Budget_and_Sales.csv' in the same directory as this script
        df = pd.read_csv('Advertising_Budget_and_Sales.csv')
        print("✅ Dataset loaded successfully!")
        print(f"📊 Dataset shape: {df.shape}")
        
        # Check current column names
        print(f"📋 Original columns: {list(df.columns)}")

        # Rename columns for better understanding based on actual number of columns
        if df.shape[1] == 4:
            df.columns = ['TV_Budget', 'Radio_Budget', 'Newspaper_Budget', 'Sales']
        elif df.shape[1] == 5:
            # If there's an index column or ID column, drop it or rename appropriately
            if df.columns[0].lower() in ['unnamed: 0', 'index', 'id']:
                df = df.drop(df.columns[0], axis=1)
                df.columns = ['TV_Budget', 'Radio_Budget', 'Newspaper_Budget', 'Sales']
            else:
                df.columns = ['Index', 'TV_Budget', 'Radio_Budget', 'Newspaper_Budget', 'Sales']
                df = df.drop('Index', axis=1)  # Drop the index column
        
        print(f"✅ Column names updated: {list(df.columns)}")
        print(f"📊 Final dataset shape: {df.shape}")

    except FileNotFoundError:
        print("❌ Error: 'Advertising_Budget_and_Sales.csv' not found!")
        print("📥 Please download the dataset from: https://www.kaggle.com/datasets/yasserh/advertising-sales-dataset")
        print("📁 Place the CSV file in the same directory as this Python script")
        return

    # 2. Exploratory Data Analysis
    print("\n2. 🔍 EXPLORATORY DATA ANALYSIS")
    print("-" * 40)

    # Basic information about the dataset
    print("📋 Dataset Overview:")
    print(df.head())
    print("\n📊 Statistical Summary:")
    print(df.describe())

    # Check for missing values
    missing_values = df.isnull().sum()
    print(f"\n❓ Missing values:\n{missing_values}")

    if missing_values.sum() == 0:
        print("✅ No missing values found!")

    # Correlation analysis
    correlation_matrix = df.corr()
    print("\n📈 Correlation with Sales:")
    sales_corr = correlation_matrix['Sales'].sort_values(ascending=False)
    for feature, corr in sales_corr.items():
        if feature != 'Sales':
            print(f"   {feature}: {corr:.4f}")

    # 3. Data Visualization
    print("\n3. 📊 CREATING VISUALIZATIONS")
    print("-" * 40)

    # Set visualization style
    plt.style.use('default')
    sns.set_palette("husl")

    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .5})
    plt.title('Correlation Matrix: Advertising Budgets vs Sales', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Scatter plots for each advertising channel
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Advertising Budget vs Sales Analysis', fontsize=16, fontweight='bold')

    # TV Budget vs Sales
    axes[0, 0].scatter(df['TV_Budget'], df['Sales'], alpha=0.6, color='blue')
    axes[0, 0].set_xlabel('TV Advertising Budget ($)')
    axes[0, 0].set_ylabel('Sales Revenue')
    axes[0, 0].set_title('TV Budget Impact on Sales')

    # Radio Budget vs Sales
    axes[0, 1].scatter(df['Radio_Budget'], df['Sales'], alpha=0.6, color='green')
    axes[0, 1].set_xlabel('Radio Advertising Budget ($)')
    axes[0, 1].set_ylabel('Sales Revenue')
    axes[0, 1].set_title('Radio Budget Impact on Sales')

    # Newspaper Budget vs Sales
    axes[1, 0].scatter(df['Newspaper_Budget'], df['Sales'], alpha=0.6, color='red')
    axes[1, 0].set_xlabel('Newspaper Advertising Budget ($)')
    axes[1, 0].set_ylabel('Sales Revenue')
    axes[1, 0].set_title('Newspaper Budget Impact on Sales')

    # Distribution of Sales
    axes[1, 1].hist(df['Sales'], bins=30, alpha=0.7, color='purple')
    axes[1, 1].set_xlabel('Sales Revenue')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Sales Revenue Distribution')

    plt.tight_layout()
    plt.savefig('advertising_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 4. Model Development
    print("\n4. 🤖 MODEL DEVELOPMENT AND TRAINING")
    print("-" * 40)

    # Prepare data for modeling
    X = df[['TV_Budget', 'Radio_Budget', 'Newspaper_Budget']]
    y = df['Sales']

    # Split data into training and testing sets (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"📊 Training set size: {X_train.shape[0]} samples")
    print(f"🧪 Testing set size: {X_test.shape[0]} samples")
    print(f"📈 Features: {list(X.columns)}")

    # Dictionary to store all models and their results
    models = {}
    results = {}

    # Model 1: Simple Linear Regression (TV Budget only)
    print("\n🎯 Training Model 1: Simple Linear Regression (TV Budget)")
    X_train_tv = X_train[['TV_Budget']]
    X_test_tv = X_test[['TV_Budget']]

    simple_model = LinearRegression()
    simple_model.fit(X_train_tv, y_train)
    y_pred_simple = simple_model.predict(X_test_tv)

    models['Simple Linear'] = simple_model
    results['Simple Linear'] = {
        'predictions': y_pred_simple,
        'r2': r2_score(y_test, y_pred_simple),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_simple)),
        'mae': mean_absolute_error(y_test, y_pred_simple)
    }

    print(f"   R² Score: {results['Simple Linear']['r2']:.4f}")
    print(f"   RMSE: {results['Simple Linear']['rmse']:.4f}")
    print(f"   Coefficient (TV): {simple_model.coef_[0]:.4f}")
    print(f"   Intercept: {simple_model.intercept_:.4f}")

    # Model 2: Multiple Linear Regression
    print("\n🎯 Training Model 2: Multiple Linear Regression (All Features)")
    multiple_model = LinearRegression()
    multiple_model.fit(X_train, y_train)
    y_pred_multiple = multiple_model.predict(X_test)

    models['Multiple Linear'] = multiple_model
    results['Multiple Linear'] = {
        'predictions': y_pred_multiple,
        'r2': r2_score(y_test, y_pred_multiple),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_multiple)),
        'mae': mean_absolute_error(y_test, y_pred_multiple)
    }

    print(f"   R² Score: {results['Multiple Linear']['r2']:.4f}")
    print(f"   RMSE: {results['Multiple Linear']['rmse']:.4f}")

    # Feature importance (coefficients)
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': multiple_model.coef_,
        'Abs_Coefficient': np.abs(multiple_model.coef_)
    }).sort_values('Abs_Coefficient', ascending=False)

    print("\n📈 Feature Importance (Coefficients):")
    for _, row in feature_importance.iterrows():
        print(f"   {row['Feature']}: {row['Coefficient']:.4f}")
    print(f"   Intercept: {multiple_model.intercept_:.4f}")

    # Model 3: Polynomial Regression
    print("\n🎯 Training Model 3: Polynomial Regression (Degree 2)")
    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.transform(X_test)

    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train)
    y_pred_poly = poly_model.predict(X_test_poly)

    models['Polynomial'] = (poly_model, poly_features)
    results['Polynomial'] = {
        'predictions': y_pred_poly,
        'r2': r2_score(y_test, y_pred_poly),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_poly)),
        'mae': mean_absolute_error(y_test, y_pred_poly)
    }

    print(f"   R² Score: {results['Polynomial']['r2']:.4f}")
    print(f"   RMSE: {results['Polynomial']['rmse']:.4f}")
    print(f"   Number of features: {X_train_poly.shape[1]}")

    # Model 4: Random Forest Regression
    print("\n🎯 Training Model 4: Random Forest Regression")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)

    models['Random Forest'] = rf_model
    results['Random Forest'] = {
        'predictions': y_pred_rf,
        'r2': r2_score(y_test, y_pred_rf),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_rf)),
        'mae': mean_absolute_error(y_test, y_pred_rf)
    }

    print(f"   R² Score: {results['Random Forest']['r2']:.4f}")
    print(f"   RMSE: {results['Random Forest']['rmse']:.4f}")

    # Random Forest feature importance
    rf_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)

    print("\n🌳 Random Forest Feature Importance:")
    for _, row in rf_importance.iterrows():
        print(f"   {row['Feature']}: {row['Importance']:.4f}")

    # 5. Model Comparison
    print("\n5. 📊 MODEL PERFORMANCE COMPARISON")
    print("-" * 40)

    # Create comparison dataframe
    model_comparison = pd.DataFrame({
        'Model': list(results.keys()),
        'R² Score': [results[model]['r2'] for model in results.keys()],
        'RMSE': [results[model]['rmse'] for model in results.keys()],
        'MAE': [results[model]['mae'] for model in results.keys()]
    })

    print("🏆 MODEL PERFORMANCE COMPARISON:")
    print(model_comparison.round(4).to_string(index=False))

    # Identify best model
    best_model_idx = model_comparison['R² Score'].idxmax()
    best_model_name = model_comparison.iloc[best_model_idx]['Model']
    print(f"\n🥇 Best Performing Model: {best_model_name}")
    print(f"   R² Score: {model_comparison.iloc[best_model_idx]['R² Score']:.4f}")

    # 6. Visualization of model predictions
    print("\n6. 📈 VISUALIZING MODEL PREDICTIONS")
    print("-" * 40)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Predictions vs Actual Values', fontsize=16, fontweight='bold')

    model_names = list(results.keys())
    for i, model_name in enumerate(model_names):
        row = i // 2
        col = i % 2
        ax = axes[row, col]

        predictions = results[model_name]['predictions']
        r2_val = results[model_name]['r2']

        ax.scatter(y_test, predictions, alpha=0.6)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax.set_xlabel('Actual Sales')
        ax.set_ylabel('Predicted Sales')
        ax.set_title(f'{model_name} Model')
        ax.text(0.05, 0.95, f'R² = {r2_val:.3f}', transform=ax.transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig('model_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 7. Business Insights and Revenue Optimization
    print("\n7. 💰 BUSINESS INSIGHTS AND REVENUE OPTIMIZATION")
    print("-" * 40)

    # Use the best performing model for business insights
    best_model = multiple_model  # Multiple Linear Regression typically performs well

    # Analyze coefficient impact
    coefficients = pd.DataFrame({
        'Channel': ['TV', 'Radio', 'Newspaper'],
        'Coefficient': best_model.coef_,
        'Impact_per_Dollar': best_model.coef_
    })

    print("📊 Sales Impact per Dollar Spent on Advertising:")
    for _, row in coefficients.iterrows():
        print(f"   {row['Channel']}: ${row['Impact_per_Dollar']:.4f} in sales per $1 spent")

    # ROI Analysis
    print(f"\n🎯 Return on Investment (ROI) Ranking:")
    roi_ranking = coefficients.sort_values('Impact_per_Dollar', ascending=False)
    for idx, (_, row) in enumerate(roi_ranking.iterrows(), 1):
        print(f"   {idx}. {row['Channel']}: {row['Impact_per_Dollar']:.4f} sales per dollar")

    # 8. Scenario Analysis for Budget Optimization
    print("\n8. 📈 SCENARIO ANALYSIS FOR BUDGET OPTIMIZATION")
    print("-" * 40)

    # Create different budget scenarios
    scenarios = pd.DataFrame({
        'Scenario': ['Current Average', 'High TV Focus', 'Balanced Approach', 'Radio Intensive'],
        'TV_Budget': [df['TV_Budget'].mean(), 300, 200, 150],
        'Radio_Budget': [df['Radio_Budget'].mean(), 50, 150, 250],
        'Newspaper_Budget': [df['Newspaper_Budget'].mean(), 25, 100, 50]
    })

    # Predict sales for each scenario
    scenarios['Predicted_Sales'] = best_model.predict(scenarios[['TV_Budget', 'Radio_Budget', 'Newspaper_Budget']])
    scenarios['Total_Budget'] = scenarios['TV_Budget'] + scenarios['Radio_Budget'] + scenarios['Newspaper_Budget']
    scenarios['Sales_per_Dollar'] = scenarios['Predicted_Sales'] / scenarios['Total_Budget']

    print("🎯 Budget Optimization Scenarios:")
    for _, row in scenarios.iterrows():
        print(f"\n{row['Scenario']}:")
        print(f"   TV: ${row['TV_Budget']:.0f}, Radio: ${row['Radio_Budget']:.0f}, Newspaper: ${row['Newspaper_Budget']:.0f}")
        print(f"   Total Budget: ${row['Total_Budget']:.0f}")
        print(f"   Predicted Sales: ${row['Predicted_Sales']:.2f}")
        print(f"   Efficiency: ${row['Sales_per_Dollar']:.4f} sales per dollar")

    # Best scenario
    best_scenario = scenarios.loc[scenarios['Sales_per_Dollar'].idxmax()]
    print(f"\n🏆 Most Efficient Scenario: {best_scenario['Scenario']}")
    print(f"   Sales per Dollar: ${best_scenario['Sales_per_Dollar']:.4f}")

    # 9. Save Results
    print("\n9. 💾 SAVING RESULTS")
    print("-" * 40)

    # Save model comparison to CSV
    model_comparison.to_csv('model_comparison_results.csv', index=False)
    print("✅ Model comparison saved to 'model_comparison_results.csv'")

    # Save scenarios to CSV
    scenarios.to_csv('budget_optimization_scenarios.csv', index=False)
    print("✅ Budget scenarios saved to 'budget_optimization_scenarios.csv'")

    # Save feature importance to CSV
    feature_importance.to_csv('feature_importance.csv', index=False)
    print("✅ Feature importance saved to 'feature_importance.csv'")

    print("\n" + "=" * 60)
    print("🎉 ANALYSIS COMPLETE!")
    print("=" * 60)
    print("📋 Summary:")
    print(f"   • Best Model: {best_model_name}")
    print(f"   • Best R² Score: {model_comparison.iloc[best_model_idx]['R² Score']:.4f}")
    print(f"   • Most Important Feature: {feature_importance.iloc[0]['Feature']}")
    print(f"   • Most Efficient Budget Scenario: {best_scenario['Scenario']}")
    print("\n📁 Generated Files:")
    print("   • correlation_heatmap.png")
    print("   • advertising_analysis.png")
    print("   • model_predictions.png")
    print("   • model_comparison_results.csv")
    print("   • budget_optimization_scenarios.csv")
    print("   • feature_importance.csv")

if __name__ == "__main__":
    main()
