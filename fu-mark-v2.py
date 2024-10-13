import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
import streamlit as st
from sklearn.impute import SimpleImputer
import warnings
import base64
import matplotlib.pyplot as plt
from datetime import datetime
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression

# New imports for Google Drive integration
import os
import io
import csv
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials

## PART 1

# Custom CSS to style the app
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;700&display=swap');
    
    html, body, [class*="css"], [class*="st-"], .stMarkdown, .stText, .stTable {
        font-family: 'Montserrat', sans-serif !important;
    }
    
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        transition: background-color 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #45a049;
    }
    
    .stSelectbox [data-baseweb="select"] {
        background-color: white;
        border-radius: 5px;
    }
    
    .stExpander {
        background-color: white;
        border-radius: 5px;
        border: 1px solid #e0e0e0;
    }
    
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    
    .sidebar .sidebar-content .sidebar-menu {
        border-top: 1px solid #e0e0e0;
        border-bottom: 1px solid #e0e0e0;
        padding: 1rem 0;
        margin: 1rem 0;
    }
    
    .sidebar .sidebar-content .sidebar-menu .stRadio > div {
        padding: 0.5rem 0;
        cursor: pointer;
        transition: background-color 0.3s ease;
        font-size: 12px !important;
    }
    
    .sidebar .sidebar-content .sidebar-menu .stRadio > div:hover {
        background-color: #e0e0e0;
    }
    
    .sidebar .sidebar-content .sidebar-menu .stRadio > div[data-checked="true"] {
        font-weight: bold;
    }
    
    .logo-img {
        display: block;
        margin: 20px auto;
        width: 100px;
        position: sticky;
        top: 0;
        z-index: 999;
    }
    </style>
    """, unsafe_allow_html=True)

# Helper functions
def add_logo(logo_path):
    try:
        with open(logo_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        st.sidebar.markdown(
            f"""
            <img src="data:image/png;base64,{encoded_string}" class="logo-img">
            """,
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        st.sidebar.warning("Logo file not found. Please check the file path.")

def styled_metric(label, value):
    return f"**{label}:** {value:.2f}"

def calculate_gap(value):
    gap = max(75 - value, 0)
    return gap * 0.25

def color_metric(label, value, is_percentage=True, reverse=False):
    try:
        value_float = float(value)
        color = "green" if (value_float > 0 and not reverse) or (value_float < 0 and reverse) else "red"
        formatted_value = f"{value_float:.2f}{'%' if is_percentage else ''}"
    except ValueError:
        # If value can't be converted to float, return it as is without color
        return f"{label}: **{value}**"
    
    return f"{label}: **:{color}[{formatted_value}]**"

def calculate_combined_score(X, y):
    correlations = np.abs(pd.DataFrame(X).corrwith(y))
    linear_model = LinearRegression().fit(X, y)
    linear_coeffs = np.abs(linear_model.coef_)
    combined_scores = correlations * linear_coeffs
    return pd.Series(combined_scores, index=X.columns)

def calculate_metrics(y_true, y_pred):
    if isinstance(y_pred[0], float):
        y_pred = (y_pred > 0.5).astype(int)
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'F1 Score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }

def select_stores(df, selector_key="store_selector"):
    unique_stores = df['establishment_name'].unique()
    selected_stores = st.multiselect('Select store(s):', unique_stores, key=selector_key)
    return selected_stores

def filter_dataframe(df, selected_stores):
    return df[df['establishment_name'].isin(selected_stores)]

# Initialize models and combined scores
def train_models(X_scaled, y_imputed, targets):
    models = {}
    combined_scores = {}
    
    for target in targets:
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_imputed[target], test_size=0.2, random_state=42)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            logistic_model = LogisticRegression(max_iter=1000, C=1.0)
            logistic_model.fit(X_train, y_train)
            
            if any(isinstance(warn.message, ConvergenceWarning) for warn in w):
                st.warning("Logistic Regression may not have converged. Consider increasing max_iter.")
        
        tree_model = DecisionTreeClassifier(random_state=42, max_depth=5, min_samples_split=10)
        tree_model.fit(X_train, y_train)
        
        models[target] = {'Logistic Regression': logistic_model, 'Decision Tree': tree_model}
        
        # Calculate combined score for this target
        combined_scores[target] = calculate_combined_score(X_scaled, y_imputed[target])
    
    return models, combined_scores

# Model Performance Comparison
def display_model_performance_comparison(X_scaled, y_imputed, models, targets, indicator_mapping):
    with st.expander("Model Performance Comparison"):
        st.write("### Model Performance Comparison")
        
        # Display explanations directly
        st.markdown("""
        **Accuracy:** The proportion of correct predictions (both true positives and true negatives) among the total number of cases examined.

        **Precision:** The proportion of true positive predictions out of all positive predictions. It answers "Of all the instances the model predicted as positive, how many were actually positive?"

        **Recall:** The proportion of true positive predictions out of all actual positive instances. It answers "Of all the actual positive instances, how many did the model correctly identify?"

        **F1 Score:** The harmonic mean of precision and recall, providing a single score that balances both metrics.

        **Which metric to prioritize?**

            - If missing positive cases is very costly, prioritize recall.
            - If false positives are very costly, prioritize precision.
            - If you need a balance between precision and recall, consider the F1 score.
            - If overall correctness is most important, focus on accuracy.
                
        """)

        for target in targets:
            st.write(f"#### {indicator_mapping[target]}")
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_imputed[target], test_size=0.2, random_state=42)
            
            results = {}
            
            # Logistic Regression
            log_pred = models[target]['Logistic Regression'].predict(X_test)
            results['Logistic Regression'] = calculate_metrics(y_test, log_pred)
            
            # Decision Tree
            tree_pred = models[target]['Decision Tree'].predict(X_test)
            results['Decision Tree'] = calculate_metrics(y_test, tree_pred)
            
            # Combined CORREL-LINEST
            combined_model = LinearRegression().fit(X_train, y_train)
            combined_pred = (combined_model.predict(X_test) > 0.5).astype(int)
            results['Combined CORREL-LINEST'] = calculate_metrics(y_test, combined_pred)
            
            # Display results
            df_results = pd.DataFrame(results).T
            st.write(df_results.round(4))
            
            # Determine the best model
            accuracies = df_results['Accuracy']
            eligible_models = accuracies[accuracies >= 0.65].index

            if not eligible_models.empty:
                # Now, among eligible_models, find the one with highest Precision
                precisions = df_results.loc[eligible_models, 'Precision']
                best_model = precisions.idxmax()
                st.write(f"Best model (highest Precision with Accuracy >= 0.65): **{best_model}**")
            else:
                st.write("No model meets the criteria of Accuracy >= 0.65.")
            
            st.write("---")

## Part 2: Analysis Functions and Main Application Logic

# Analysis functions

def top_down_analysis(X_scaled, y_imputed, models, combined_scores, scaler, targets, indicator_mapping, X_imputed_data, button_key):
    st.write("## Top-Down Analysis")
    
    # Display current percentages and input desired goals
    st.write("### Current and Desired OKR Percentages:")
    current_percentages = {}
    desired_goals = {}
    cols = st.columns(len(targets))
    for idx, target in enumerate(targets):
        current_percentage = (y_imputed[target] == 1).mean() * 100
        current_percentages[target] = current_percentage
        with cols[idx]:
            st.metric(
                label=indicator_mapping[target],
                value=f"{current_percentage:.2f}%",
                delta=None
            )
            desired_goals[target] = st.number_input(
                f"Desired Goal for {indicator_mapping[target]} (%)", 
                min_value=0.0, 
                max_value=100.0, 
                value=float(current_percentage), 
                step=0.1, 
                key=f"goal_{button_key}_{target}"
            )
    st.write("---")
    
    # Add Model Performance Comparison
    display_model_performance_comparison(X_scaled, y_imputed, models, targets, indicator_mapping)
    
    # Select model
    model_choice = st.selectbox(
        f'Select Model for Analysis ({button_key})', 
        ('Logistic Regression', 'Decision Tree', 'Combined CORREL-LINEST'), 
        key=f"model_choice_{button_key}"
    )
    
    if st.button('Predict Required KPI Changes', key=f"predict_button_{button_key}"):
        # Calculate required changes for each target
        required_changes = {}
        for target in targets:
            required_change = desired_goals[target] - current_percentages[target]
            # Ensure that required change does not exceed bounds
            required_change = min(max(required_change, -current_percentages[target]), 100 - current_percentages[target])
            required_changes[target] = required_change
        
        # Get feature importances
        if model_choice == 'Logistic Regression':
            importances = pd.Series(np.abs(models['satisfaction']['Logistic Regression'].coef_[0]), index=X_scaled.columns)
        elif model_choice == 'Decision Tree':
            importances = pd.Series(models['satisfaction']['Decision Tree'].feature_importances_, index=X_scaled.columns)
        else:  # Combined CORREL-LINEST
            importances = combined_scores['satisfaction']
        
        sorted_features = importances.sort_values(ascending=False)
        
        st.write("### Required KPI Changes:")
        for feature in sorted_features.index:
            current_percentage = (X_imputed_data[feature] > X_imputed_data[feature].mean()).mean() * 100
            
            # Calculate required change (simplified proportional allocation)
            required_change = sum(required_changes.values()) * (importances[feature] / importances.sum())
            
            # Ensure that required_change does not push current_percentage over 100%
            max_possible_change = 100 - current_percentage
            min_possible_change = -current_percentage
            required_change = min(max(required_change, min_possible_change), max_possible_change)
            
            st.write(f"#### {feature}")
            cols_feature = st.columns(3)
            with cols_feature[0]:
                st.write(f"**Importance:** {importances[feature]:.4f}")
            with cols_feature[1]:
                st.write(f"**Current:** {current_percentage:.2f}%")
            with cols_feature[2]:
                st.write(f"**Required Change:** {required_change:.2f} pp")
            st.write("---")

def display_top_7_kpis(X_scaled, y_imputed, models, combined_scores, scaler, targets, indicator_mapping, kpis, X_imputed_data):
    st.write("## Top 7 KPIs Analysis")
    
    # Display Target Metrics Summary
    st.write("### Target Metrics Summary:")
    metrics_cols = st.columns(len(targets))
    for idx, target in enumerate(targets):
        current_percentage = (y_imputed[target] == 1).mean() * 100
        metrics_cols[idx].metric(
            label=indicator_mapping[target],
            value=f"{current_percentage:.2f}%",
            delta=None
        )
    st.write("---")
    
    # Add Model Performance Comparison
    display_model_performance_comparison(X_scaled, y_imputed, models, targets, indicator_mapping)
    
    # Select model
    model_choice = st.selectbox(
        'Select Model for KPI Ranking', 
        ('Logistic Regression', 'Decision Tree', 'Combined CORREL-LINEST'),
        key="kpi_ranking_model_choice_top7"
    )
    
    if model_choice == 'Logistic Regression':
        importances = pd.Series(np.abs(models['satisfaction']['Logistic Regression'].coef_[0]), index=X_scaled.columns)
    elif model_choice == 'Decision Tree':
        importances = pd.Series(models['satisfaction']['Decision Tree'].feature_importances_, index=X_scaled.columns)
    else:  # Combined CORREL-LINEST
        importances = combined_scores['satisfaction']
    
    sorted_features = importances.sort_values(ascending=False)
    
    st.write("## Enter desired changes:")
    changes = {}
    for feature in sorted_features.index[:7]:  # Top 7 features
        current_percentage = (X_imputed_data[feature] > X_imputed_data[feature].mean()).mean() * 100
        st.write(f"### {feature}")
        cols_feature = st.columns(3)
        with cols_feature[0]:
            st.write(f"**Importance:** {importances[feature]:.4f}")
        with cols_feature[1]:
            st.write(f"**Current:** {current_percentage:.2f}%")
            max_possible_increase = 100.0 - current_percentage
            max_possible_decrease = -current_percentage
            changes[feature] = st.number_input(
                f"Change for {feature} (pp)", 
                value=0.0, 
                min_value=max_possible_decrease,
                max_value=max_possible_increase,
                step=0.1, 
                format="%.1f", 
                key=f"change_top7_{feature}"
            )
        with cols_feature[2]:
            new_percentage = current_percentage + changes[feature]
            new_percentage = min(max(new_percentage, 0.0), 100.0)
            new_gap = calculate_gap(new_percentage)
            st.write("**After Change:**")
            st.markdown(color_metric("New %", new_percentage, reverse=True), unsafe_allow_html=True)
            st.markdown(color_metric("New Gap", new_gap, is_percentage=False), unsafe_allow_html=True)
        st.write("---")
    
    if st.button('Predict', key='predict_top7_kpis'):
        X_modified_imputed = X_imputed_data.copy()
        any_changes = False
        debug_info = {}  # For storing debug information

        for feature in sorted_features.index[:7]:  # Top 7 features
            change = changes.get(feature, 0)
            if change != 0:
                any_changes = True
                # Adjust the feature values to reach the desired mean
                current_mean = X_imputed_data[feature].mean()
                desired_mean = current_mean + (change / 100)
                desired_mean = min(max(desired_mean, 0), 1)  # Ensure it's between 0 and 1
                adjustment = desired_mean - current_mean
                X_modified_imputed[feature] += adjustment
                X_modified_imputed[feature] = X_modified_imputed[feature].clip(0, 1)
        
            # Store debug information
            debug_info[feature] = {
                'original': X_imputed_data[feature].mean(),
                'modified': X_modified_imputed[feature].mean(),
                'change': change
            }

        # Display debug information in an expander
        with st.expander("🔍 View Debug Information"):
            st.write("**Changes Applied to Features:**")
            for feature, info in debug_info.items():
                st.write(f"**{feature}**:")
                st.write(f"  - Original Mean: {info['original']:.4f}")
                st.write(f"  - Modified Mean: {info['modified']:.4f}")
                st.write(f"  - Change (pp): {info['change']:.1f}")
                st.write("---")

        if any_changes:
            X_modified_scaled = pd.DataFrame(scaler.transform(X_modified_imputed), columns=X_imputed_data.columns)
            for target in targets:
                st.markdown(f"### **{indicator_mapping[target]} Results:**")
                st.write("")  # Add space

                current_percentage = (y_imputed[target] == 1).mean() * 100

                for method in ['Logistic Regression', 'Decision Tree', 'Combined CORREL-LINEST']:
                    st.markdown(f"**:blue[{method}:]**")

                    if method in ['Logistic Regression', 'Decision Tree']:
                        model = models[target][method]
                        predictions = model.predict_proba(X_modified_scaled)[:, 1]
                    else:  # Combined CORREL-LINEST
                        combined_model = LinearRegression().fit(X_scaled, y_imputed[target])
                        predictions = combined_model.predict(X_modified_scaled)
                        predictions = np.clip(predictions, 0, 1)

                    predicted_value = np.mean(predictions) * 100
                    predicted_value = min(max(predicted_value, 0.0), 100.0)
                    predicted_change = predicted_value - current_percentage
                    final_predicted = predicted_value

                    # Ensure final_predicted does not exceed 100%
                    final_predicted = min(max(final_predicted, 0.0), 100.0)
                    predicted_change = final_predicted - current_percentage

                    # Highlight the metrics using Streamlit's metric component
                    st.metric("Current (%)", f"{current_percentage:.2f}%", delta=None)
                    st.metric("Predicted Change (pp)", f"{predicted_change:.2f} pp", delta=predicted_change)
                    st.metric("Final Predicted (%)", f"{final_predicted:.2f}%", delta=predicted_change)

                    st.write("---")  # Separator between methods

                st.write("---")  # Separator between indicators
        else:
            st.info("No changes were made. The prediction remains the same as the current percentage.")

def display_all_kpis(X_scaled, y_imputed, models, combined_scores, scaler, targets, indicator_mapping, kpis, X_imputed_data):
    st.write("## All KPIs Analysis")
    
    # Display Target Metrics Summary
    st.write("### Target Metrics Summary:")
    metrics_cols = st.columns(len(targets))
    for idx, target in enumerate(targets):
        current_percentage = (y_imputed[target] == 1).mean() * 100
        metrics_cols[idx].metric(
            label=indicator_mapping[target],
            value=f"{current_percentage:.2f}%",
            delta=None
        )
    st.write("---")
    
    # Add Model Performance Comparison
    display_model_performance_comparison(X_scaled, y_imputed, models, targets, indicator_mapping)
    
    # Select model for KPI Ranking
    model_choice = st.selectbox(
        'Select Model for KPI Ranking', 
        ('Logistic Regression', 'Decision Tree', 'Combined CORREL-LINEST'),
        key="kpi_ranking_model_choice_all"
    )
    
    if model_choice == 'Logistic Regression':
        importances = pd.Series(np.abs(models['satisfaction']['Logistic Regression'].coef_[0]), index=X_scaled.columns)
    elif model_choice == 'Decision Tree':
        importances = pd.Series(models['satisfaction']['Decision Tree'].feature_importances_, index=X_scaled.columns)
    else:  # Combined CORREL-LINEST
        importances = combined_scores['satisfaction']
    
    sorted_features = importances.sort_values(ascending=False)
    
    st.write("## Enter desired changes:")
    changes = {}
    for feature in sorted_features.index:  # Iterate over all features
        current_percentage = (X_imputed_data[feature] > X_imputed_data[feature].mean()).mean() * 100
        st.write(f"### {feature}")
        cols_feature = st.columns(3)
        with cols_feature[0]:
            st.write(f"**Importance:** {importances[feature]:.4f}")
        with cols_feature[1]:
            st.write(f"**Current:** {current_percentage:.2f}%")
            max_possible_increase = 100.0 - current_percentage
            max_possible_decrease = -current_percentage
            changes[feature] = st.number_input(
                f"Change for {feature} (pp)", 
                value=0.0, 
                min_value=max_possible_decrease,
                max_value=max_possible_increase,
                step=0.1, 
                format="%.1f", 
                key=f"change_all_{feature}"
            )
        with cols_feature[2]:
            new_percentage = current_percentage + changes[feature]
            new_percentage = min(max(new_percentage, 0.0), 100.0)
            new_gap = calculate_gap(new_percentage)
            st.write("**After Change:**")
            st.markdown(color_metric("New %", new_percentage, reverse=True), unsafe_allow_html=True)
            st.markdown(color_metric("New Gap", new_gap, is_percentage=False), unsafe_allow_html=True)
        st.write("---")
    
    if st.button('Predict', key='predict_all_kpis'):
        X_modified_imputed = X_imputed_data.copy()
        any_changes = False
        debug_info = {}  # For storing debug information

        for feature in sorted_features.index:  # Iterate over all features
            change = changes.get(feature, 0)
            if change != 0:
                any_changes = True
                # Adjust the feature values to reach the desired mean
                current_mean = X_imputed_data[feature].mean()
                desired_mean = current_mean + (change / 100)
                desired_mean = min(max(desired_mean, 0), 1)  # Ensure it's between 0 and 1
                adjustment = desired_mean - current_mean
                X_modified_imputed[feature] += adjustment
                X_modified_imputed[feature] = X_modified_imputed[feature].clip(0, 1)
        
            # Store debug information
            debug_info[feature] = {
                'original': X_imputed_data[feature].mean(),
                'modified': X_modified_imputed[feature].mean(),
                'change': change
            }

        # Display debug information in an expander
        with st.expander("🔍 View Debug Information"):
            st.write("**Changes Applied to Features:**")
            for feature, info in debug_info.items():
                st.write(f"**{feature}**:")
                st.write(f"  - Original Mean: {info['original']:.4f}")
                st.write(f"  - Modified Mean: {info['modified']:.4f}")
                st.write(f"  - Change (pp): {info['change']:.1f}")
                st.write("---")

        if any_changes:
            X_modified_scaled = pd.DataFrame(scaler.transform(X_modified_imputed), columns=X_imputed_data.columns)
            for target in targets:
                st.markdown(f"### **{indicator_mapping[target]} Results:**")
                st.write("")  # Add space

                current_percentage = (y_imputed[target] == 1).mean() * 100

                for method in ['Logistic Regression', 'Decision Tree', 'Combined CORREL-LINEST']:
                    st.markdown(f"**:blue[{method}:]**")

                    if method in ['Logistic Regression', 'Decision Tree']:
                        model = models[target][method]
                        predictions = model.predict_proba(X_modified_scaled)[:, 1]
                    else:  # Combined CORREL-LINEST
                        combined_model = LinearRegression().fit(X_scaled, y_imputed[target])
                        predictions = combined_model.predict(X_modified_scaled)
                        predictions = np.clip(predictions, 0, 1)

                    predicted_value = np.mean(predictions) * 100
                    predicted_value = min(max(predicted_value, 0.0), 100.0)
                    predicted_change = predicted_value - current_percentage
                    final_predicted = predicted_value

                    # Ensure final_predicted does not exceed 100%
                    final_predicted = min(max(final_predicted, 0.0), 100.0)
                    predicted_change = final_predicted - current_percentage

                    # Highlight the metrics using Streamlit's metric component
                    st.metric("Current (%)", f"{current_percentage:.2f}%", delta=None)
                    st.metric("Predicted Change (pp)", f"{predicted_change:.2f} pp", delta=predicted_change)
                    st.metric("Final Predicted (%)", f"{final_predicted:.2f}%", delta=predicted_change)

                    st.write("---")  # Separator between methods

                st.write("---")  # Separator between indicators
        else:
            st.info("No changes were made. The prediction remains the same as the current percentage.")

def analyze_category_impact(X_scaled, X_imputed, y_imputed, models, combined_scores, categories, kpis, scaler, targets, indicator_mapping):
    st.write("## Category Impact Analysis")
    
    # Display Target Metrics Summary
    st.write("### Target Metrics Summary:")
    metrics_cols = st.columns(len(targets))
    for idx, target in enumerate(targets):
        current_percentage = (y_imputed[target] == 1).mean() * 100
        metrics_cols[idx].metric(
            label=indicator_mapping[target],
            value=f"{current_percentage:.2f}%",
            delta=None
        )
    st.write("---")

    # Add Model Performance Comparison
    display_model_performance_comparison(X_scaled, y_imputed, models, targets, indicator_mapping)

    # Calculate per feature importances
    # Correlation
    corr = np.abs(pd.DataFrame(X_scaled).corrwith(y_imputed['satisfaction']))
    corr.index = X_scaled.columns

    # Linear Regression Coefficient
    linreg_model = LinearRegression().fit(X_scaled, y_imputed['satisfaction'])
    linreg_coef = pd.Series(np.abs(linreg_model.coef_), index=X_scaled.columns)

    # Combined Score
    combined_score = corr * linreg_coef

    # Logistic Regression Coefficient
    logistic_model = models['satisfaction']['Logistic Regression']
    logistic_coef = pd.Series(np.abs(logistic_model.coef_[0]), index=X_scaled.columns)

    # Decision Tree Feature Importance
    tree_model = models['satisfaction']['Decision Tree']
    tree_importance = pd.Series(tree_model.feature_importances_, index=X_scaled.columns)

    # Create DataFrame
    feature_importances_df = pd.DataFrame({
        'Correlation': corr,
        'LinReg_Coefficient': linreg_coef,
        'Combined_Score': combined_score,
        'Logistic_Coefficient': logistic_coef,
        'DecisionTree_Importance': tree_importance
    })

    # Add 'Category' column
    feature_importances_df['Category'] = categories.reindex(X_scaled.columns)
    
    # Drop NaN categories (e.g., 'Year', 'month', 'establishment_name')
    feature_importances_df = feature_importances_df.dropna(subset=['Category'])

    # Group by 'Category' and sum
    category_importances_df = feature_importances_df.groupby('Category').sum()

    # Rename columns
    category_importances_df = category_importances_df.rename(columns={
        'LinReg_Coefficient': 'Impact',
        'Combined_Score': 'Combined CORREL-LINEST',
        'Logistic_Coefficient': 'Logistic Regression Importance',
        'DecisionTree_Importance': 'Decision Tree Importance'
    })

    # Rearrange columns
    category_importances_df = category_importances_df[['Correlation', 'Impact', 'Combined CORREL-LINEST',
                                                      'Logistic Regression Importance', 'Decision Tree Importance']]

    # Format numbers to 4 decimal places
    category_importances_df = category_importances_df.round(4)

    # Sort categories by 'Combined CORREL-LINEST' in descending order
    category_importances_df = category_importances_df.sort_values('Combined CORREL-LINEST', ascending=False)

    # Display the table
    st.write("### Category Importances")
    st.dataframe(category_importances_df)

    # Category-based prediction
    st.write("## Category-based Prediction")
    changes = {}
    for category in category_importances_df.index:
        st.write(f"### {category}")
        cols_feature = st.columns(2)
        with cols_feature[0]:
            st.write("**Category Features:**")
            category_features = categories[categories == category].index.tolist()
            category_features = [feat for feat in category_features if feat in X_imputed.columns]
            st.write(category_features)
            # For binary features (0/1), current percentage is mean * 100
            if category_features:
                current_percentage = X_imputed[category_features].mean().mean() * 100
                st.write(f"**Current:** {current_percentage:.2f}%")
            else:
                current_percentage = 0.0
                st.write(f"**Current:** {current_percentage:.2f}%")
            max_possible_increase = 100.0 - current_percentage
            max_possible_decrease = -current_percentage
            changes[category] = st.number_input(
                f"Change for {category} (pp)", 
                value=0.0, 
                min_value=max_possible_decrease,
                max_value=max_possible_increase,
                step=0.1, 
                format="%.1f", 
                key=f"change_category_{category}"
            )
        with cols_feature[1]:
            new_percentage = current_percentage + changes[category]
            new_percentage = min(max(new_percentage, 0.0), 100.0)
            new_gap = calculate_gap(new_percentage)
            st.write("**After Change:**")
            st.metric("New %", f"{new_percentage:.2f}%", delta=None)
    
    if st.button('Predict Category Impact', key='predict_category_impact'):
        X_modified_imputed = X_imputed.copy()
        any_changes = False
        for category, change in changes.items():
            if change != 0:
                any_changes = True
                category_features = categories[categories == category].index.tolist()
                category_features = [feat for feat in category_features if feat in X_imputed.columns]
                # Adjust the mean of the features in the category by the change
                for feat in category_features:
                    current_mean = X_imputed[feat].mean()
                    desired_mean = current_mean + (change / 100)
                    desired_mean = min(max(desired_mean, 0), 1)  # Ensure it's between 0 and 1
                    # Adjust the feature values to reach the desired mean
                    adjustment = desired_mean - current_mean
                    X_modified_imputed[feat] += adjustment
                    X_modified_imputed[feat] = X_modified_imputed[feat].clip(0, 1)

        if any_changes:
            # Re-standardize the modified features
            X_modified_scaled = pd.DataFrame(scaler.transform(X_modified_imputed), columns=X_imputed.columns)
            for target in targets:
                st.markdown(f"### **{indicator_mapping[target]} Results:**")
                
                current_percentage = (y_imputed[target] == 1).mean() * 100
                
                for method in ['Logistic Regression', 'Decision Tree', 'Combined CORREL-LINEST']:
                    st.markdown(f"**:blue[{method}:]**")
                    
                    if method in ['Logistic Regression', 'Decision Tree']:
                        model = models[target][method]
                        predictions = model.predict_proba(X_modified_scaled)[:, 1]
                    else:  # Combined CORREL-LINEST
                        combined_model = LinearRegression().fit(X_scaled, y_imputed[target])
                        predictions = combined_model.predict(X_modified_scaled)
                        predictions = np.clip(predictions, 0, 1)
                    
                    predicted_value = np.mean(predictions) * 100
                    predicted_value = min(max(predicted_value, 0.0), 100.0)
                    predicted_change = predicted_value - current_percentage
                    final_predicted = predicted_value

                    # Ensure final_predicted does not exceed 100%
                    final_predicted = min(max(final_predicted, 0.0), 100.0)
                    predicted_change = final_predicted - current_percentage

                    # Highlight the metrics using Streamlit's metric component
                    st.metric("Current (%)", f"{current_percentage:.2f}%", delta=None)
                    st.metric("Predicted Change (pp)", f"{predicted_change:.2f} pp", delta=predicted_change)
                    st.metric("Final Predicted (%)", f"{final_predicted:.2f}%", delta=predicted_change)
                    
                    st.write("---")  # Separator between methods

                st.write("---")  # Separator between indicators
        else:
            st.info("No changes were made. The prediction remains the same as the current percentage.")

def display_individual_correlation_matrix(df, X_imputed, display_correlation_matrix_func):
    st.title("Individual Store: Correlation Matrix")
    
    # Selector for individual stores
    selected_stores = select_stores(df, selector_key="individual_correlation_matrix_store_selector")
    
    if selected_stores:
        df_filtered = filter_dataframe(df, selected_stores)
        # Extract feature columns
        feature_columns = [col for col in df_filtered.columns if col not in ['establishment_name', 'month', 'Year'] + ['satisfaction', 'value_for_money', 'return_probability', 'nps']]
        features_filtered = df_filtered[feature_columns]
        
        # Convert to numeric
        features_converted = features_filtered.apply(pd.to_numeric, errors='coerce')
        features_converted = features_converted.select_dtypes(include=[np.number])
        
        if features_converted.empty:
            st.error("No numeric features available for the selected store(s).")
            return
        
        # Impute missing values
        imputer = SimpleImputer(strategy='most_frequent')
        X_filtered_imputed = pd.DataFrame(imputer.fit_transform(features_converted), columns=features_converted.columns)
        
        # Display correlation matrix
        st.write("### Correlation Matrix for Selected Store(s)")
        display_correlation_matrix_func(X_filtered_imputed)
    else:
        st.warning("Please select at least one store to view its correlation matrix.")

def display_correlation_matrix(numeric_df):
    st.write("### Correlation Matrix")
    if not numeric_df.empty and numeric_df.shape[1] > 0:
        # Exclude specific columns if necessary
        columns_to_exclude = ['establishment_name', 'month', 'Year']
        correlation_matrix = numeric_df.drop(columns=columns_to_exclude, errors='ignore').corr()

        # Apply background gradient for better visualization
        styled_corr_matrix = correlation_matrix.style.background_gradient(cmap='Greens')
        st.dataframe(styled_corr_matrix)
    else:
        st.write("Correlation matrix cannot be computed because there are no numeric columns.")

def display_time_series_analysis(df, targets, indicator_mapping):
    st.title("Time-Series Analysis")

    # Check if 'month' and 'Year' columns exist
    if 'month' not in df.columns or 'Year' not in df.columns:
        st.error("The dataset must contain 'month' and 'Year' columns for Time-Series analysis.")
        return

    # Create a 'Date' column from 'month' and 'Year'
    try:
        df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['month'].astype(str), format='%Y-%m')
    except Exception as e:
        st.error(f"Error parsing 'month' and 'Year' columns into datetime: {e}")
        return

    # Sort the dataframe by Date
    df = df.sort_values('Date')

    # Ensure target variables are present
    for target in targets:
        if target not in df.columns:
            st.error(f"Target variable '{target}' not found in the dataset.")
            return

    # Allow users to select a date range
    st.sidebar.header("Time-Series Filters")
    min_date = df['Date'].min()
    max_date = df['Date'].max()

    start_date = st.sidebar.date_input("Start date", min_date, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("End date", max_date, min_value=min_date, max_value=max_date)

    if start_date > end_date:
        st.error("Start date must be before or equal to End date.")
        return

    # Allow users to select the forecasting model
    st.sidebar.header("Forecasting Model Selection")
    model_choice = st.sidebar.selectbox(
        "Select the forecasting model:",
        ("Linear Regression", "Exponential Smoothing")
    )

    # Filter the dataframe based on selected date range
    df_filtered = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))].copy()

    if df_filtered.empty:
        st.warning("No data available for the selected date range.")
        return

    # Convert target columns to numeric
    for target in targets:
        df_filtered[target] = pd.to_numeric(df_filtered[target], errors='coerce')

    # Handle missing values in target columns
    df_filtered[targets] = df_filtered[targets].fillna(0)

    # Group by 'Date' and calculate the percentage of times the target variable is 1
    df_grouped = df_filtered.groupby('Date').agg({target: 'mean' for target in targets})
    df_grouped = df_grouped * 100  # Convert to percentage

    # Display the filtered data
    st.write(f"### Data from {start_date} to {end_date}")
    st.dataframe(df_grouped.style.format("{:.2f}%"))

    st.write("### Trend Projection for Next 3 Months")

    projections = {}

    for target in targets:
        # Check if there are enough data points
        if df_grouped.shape[0] < 3:
            st.warning(f"Not enough data points to perform trend analysis for {indicator_mapping[target]}.")
            continue

        y = df_grouped[target]

        # Handle missing values
        y = y.fillna(method='ffill').fillna(method='bfill')

        # Forecasting based on selected model
        if model_choice == "Linear Regression":
            # Prepare the data for linear regression
            df_trend = df_grouped.copy()
            df_trend.reset_index(inplace=True)
            df_trend['Month_Num'] = df_trend['Date'].dt.year * 12 + df_trend['Date'].dt.month  # Numerical representation of time

            X = df_trend[['Month_Num']]
            y_values = df_trend[target]

            # Fit linear regression
            model = LinearRegression()
            model.fit(X, y_values)

            # Predict the value 3 months into the future
            last_month_num = df_trend['Month_Num'].iloc[-1]
            future_month_nums = [last_month_num + i for i in range(1, 4)]  # Next 3 months
            future_dates = [df_trend['Date'].iloc[-1] + pd.DateOffset(months=i) for i in range(1, 4)]
            predicted_values = model.predict(np.array(future_month_nums).reshape(-1, 1))

            # Convert predicted_values to a NumPy array
            predicted_values = np.array(predicted_values)

            # Model description
            model_description = "Linear Regression"

        elif model_choice == "Exponential Smoothing":
            # Fit Exponential Smoothing model
            model = ExponentialSmoothing(y, trend='add', seasonal=None, initialization_method="estimated")
            model_fit = model.fit()

            # Forecast the next 3 periods
            predicted_values = model_fit.forecast(steps=3)

            # Convert predicted_values to a NumPy array
            predicted_values = np.array(predicted_values)

            future_dates = [df_grouped.index[-1] + pd.DateOffset(months=i) for i in range(1, 4)]

            # Model description
            model_description = "Exponential Smoothing"

        else:
            st.error("Selected model is not implemented.")
            return

        # Calculate current value as the average of the past 3 months
        current_value = y.iloc[-3:].mean()

        # Calculate percentage point change
        predicted_value = predicted_values[-1]  # Prediction for 3 months ahead
        change_pp = predicted_value - current_value

        # Determine trend direction
        if change_pp > 0:
            trend_direction = 'Increasing'
            arrow_color = 'green'
        elif change_pp < 0:
            trend_direction = 'Decreasing'
            arrow_color = 'red'
        else:
            trend_direction = 'Stable'
            arrow_color = 'black'

        # Cap the predicted value between 0 and 100
        predicted_value = min(max(predicted_value, 0.0), 100.0)

        # Store the projection
        projections[target] = {
            'Current 3-Month Average (%)': current_value,
            'Predicted Value in 3 Months (%)': predicted_value,
            'Change (pp)': change_pp,
            'Trend': trend_direction
        }

        # Plot the trend with projection
        st.write(f"#### {indicator_mapping[target]} Projection")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df_grouped.index, y, marker='o', linestyle='-', label='Actual')
        ax.set_xlabel('Date')
        ax.set_ylabel(f"{indicator_mapping[target]} (%)")
        ax.set_title(f"{indicator_mapping[target]} Over Time with Projection")
        ax.grid(True)

        # Add current 3-month average as a red dot
        average_date = df_grouped.index[-1]  # Use the last date for the average point
        ax.scatter(average_date, current_value, color='red', zorder=5, label='3-Month Avg')

        # Annotate the average point
        ax.annotate(f"{current_value:.2f}%", (average_date, current_value),
                    textcoords="offset points", xytext=(0,10), ha='center', color='red')

        # Add predicted value as an orange dot
        future_date = future_dates[-1]
        predicted_value_capped = min(max(predicted_value, 0), 100)
        ax.scatter(future_date, predicted_value_capped, color='orange', zorder=5, label='Predicted Value')

        # Annotate the predicted value
        ax.annotate(f"{predicted_value_capped:.2f}%", (future_date, predicted_value_capped),
                    textcoords="offset points", xytext=(0,10), ha='center', color='orange')

        # Add projection line
        ax.plot(future_dates, predicted_values, marker='o', linestyle='--', color='orange', label='Projection')
        ax.legend()
        st.pyplot(fig)

        # Third Change: Get top 7 KPIs contributing to the trend
        # For simplicity, we can calculate the correlation of each feature with the target over the past 3 months
        recent_data = df_filtered[df_filtered['Date'] >= df_filtered['Date'].max() - pd.DateOffset(months=2)]  # Last 3 months

        # Ensure there is data in recent_data
        if not recent_data.empty:
            # Exclude targets and date columns
            feature_cols = [col for col in df.columns if col not in targets + ['Date', 'Year', 'month', 'establishment_name']]
            features_recent = recent_data[feature_cols].apply(pd.to_numeric, errors='coerce')
            features_recent = features_recent.select_dtypes(include=[np.number])
            target_recent = recent_data[target]

            # Handle missing values
            features_recent = features_recent.fillna(0)
            target_recent = target_recent.fillna(0)

            # Calculate correlations
            correlations = features_recent.corrwith(target_recent)
            correlations_abs = correlations.abs()
            top_7_features = correlations_abs.sort_values(ascending=False).head(7)

            # Wrap KPI trends inside an expander
            with st.expander(f"Top 7 KPIs impacting {indicator_mapping[target]}", expanded=False):
                st.write(f"Top 7 KPIs contributing to the {trend_direction.lower()} trend in {indicator_mapping[target]}:")
                # Add the predictive model used
                st.write(f"Predictive Model: {model_description}")

                # Show the trends of these features over the past 3 months
                for feature in top_7_features.index:
                    st.write(f"##### {feature}")
                    feature_grouped = recent_data.groupby('Date')[feature].mean()

                    fig_feat, ax_feat = plt.subplots(figsize=(10, 3))
                    ax_feat.plot(feature_grouped.index, feature_grouped.values, marker='o', linestyle='-')
                    ax_feat.set_xlabel('Date')
                    ax_feat.set_ylabel(f"{feature}")
                    ax_feat.set_title(f"{feature} Trend Over Last 3 Months")
                    ax_feat.grid(True)
                    st.pyplot(fig_feat)

                    # Show the impact of each KPI
                    kpi_correlation = correlations[feature]
                    st.write(f"Impact on {indicator_mapping[target]}: Correlation = {kpi_correlation:.2f}")

                    # Estimate pp impact (simplified estimation)
                    pp_impact = kpi_correlation * change_pp
                    st.write(f"Estimated impact on {indicator_mapping[target]} over next 3 months: {pp_impact:.2f} pp")
                    st.write("---")

    # Display the projections
    if projections:
        projection_df = pd.DataFrame(projections).T
        projection_df = projection_df[['Current 3-Month Average (%)', 'Predicted Value in 3 Months (%)', 'Change (pp)', 'Trend']]

        # Fix the ValueError by formatting only numeric columns
        numeric_columns = ['Current 3-Month Average (%)', 'Predicted Value in 3 Months (%)', 'Change (pp)']
        projection_df_style = projection_df.style.format({col: "{:.2f}" for col in numeric_columns})

        # Apply color coding to 'Change (pp)' and 'Trend' columns
        def color_negative_red(val):
            color = 'red' if val < 0 else 'green'
            return f'color: {color}'

        def highlight_trend(val):
            color = 'green' if val.lower() == 'increasing' else 'red' if val.lower() == 'decreasing' else 'black'
            return f'color: {color}'

        projection_df_style = projection_df_style.applymap(color_negative_red, subset=['Change (pp)'])
        projection_df_style = projection_df_style.applymap(highlight_trend, subset=['Trend'])

        st.table(projection_df_style)

    else:
        st.info("No projections to display.")

## Functions for Google Drive Integration

def authenticate_google_drive():
    """
    Authenticates to Google Drive using service account credentials stored in Streamlit Secrets.

    Returns:
        creds: The authenticated credentials object.
    """
    # Define the scope for Google Drive API
    SCOPES = ['https://www.googleapis.com/auth/drive.file']

    # Load the service account info from Streamlit secrets
    service_account_info = st.secrets["gcp_service_account"]

    # Create credentials using the service account info
    creds = service_account.Credentials.from_service_account_info(
        service_account_info, scopes=SCOPES
    )

    return creds

def upload_file_to_drive(file_name, file_data):
    """
    Uploads a file to Google Drive using the authenticated credentials.

    Args:
        file_name (str): The name of the file to upload.
        file_data (UploadedFile): The file data from Streamlit's file uploader.

    Returns:
        file_id (str): The ID of the uploaded file on Google Drive.
        webViewLink (str): A link to view the uploaded file.
    """
    try:
        # Authenticate and build the Google Drive service
        creds = authenticate_google_drive()
        service = build('drive', 'v3', credentials=creds)

        # Prepare file metadata and media content
        file_metadata = {'name': file_name}
        fh = io.BytesIO(file_data.getvalue())
        media = MediaIoBaseUpload(
            fh,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

        # Upload the file to Google Drive
        uploaded_file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id, webViewLink'
        ).execute()

        # Retrieve the file ID and webViewLink
        file_id = uploaded_file.get('id')
        webViewLink = uploaded_file.get('webViewLink')

        # Optionally, set file permissions to make it accessible (e.g., anyone with the link can view)
        # Uncomment the following lines if you want to set the permissions
        '''
        permission = {
            'type': 'anyone',
            'role': 'reader'
        }
        service.permissions().create(
            fileId=file_id,
            body=permission
        ).execute()
        '''

        # Log success message and display the link
        st.success(f"File '{file_name}' uploaded successfully to Google Drive.")
        st.write(f"Access the uploaded file here: [View File]({webViewLink})")

        # Return the file ID and webViewLink
        return file_id, webViewLink

    except Exception as e:
        # Handle exceptions and display error messages
        st.error(f"An error occurred during file upload: {e}")
        return None, None

def store_uploaded_file_link(file_name, link):
    import csv
    from datetime import datetime

    file_exists = os.path.isfile('uploaded_files.csv')
    with open('uploaded_files.csv', 'a', newline='') as csvfile:
        fieldnames = ['timestamp', 'file_name', 'link']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({'timestamp': datetime.now().isoformat(), 'file_name': file_name, 'link': link})

def display_past_predictions():
    import csv

    if os.path.isfile('uploaded_files.csv'):
        st.write("### Previously Uploaded Files:")
        with open('uploaded_files.csv', 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            uploaded_files = list(reader)
            if uploaded_files:
                for file_info in uploaded_files:
                    timestamp = file_info['timestamp']
                    file_name = file_info['file_name']
                    link = file_info['link']
                    st.write(f"- **{file_name}** uploaded on {timestamp}: [View File]({link})")
            else:
                st.write("No files have been uploaded yet.")
    else:
        st.write("No files have been uploaded yet.")

## PART 3

# Main code and remaining functions
def main():
    uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

    # Add logo to the sidebar
    add_logo("logo_freatz.png")  # Adjust the path as needed

    # Updated sidebar menu with the new items
    st.sidebar.title("Navigation")
    
    st.sidebar.markdown('<div class="sidebar-menu">', unsafe_allow_html=True)
    page = st.sidebar.radio("", [
        # All Stores Group
        "All Stores: 7 KPIs",
        "All Stores: All KPIs",
        "All Stores: Top-Down Analysis",
        "All Stores: Correlation Matrix",
        "All Stores: Category Impact",
        # Individual Store Group
        "Individual Store: 7 KPIs",
        "Individual Store: All KPIs",
        "Individual Store: Top-Down Analysis",
        "Individual Store: Correlation Matrix",
        "Individual Store: Category Impact",
        # Other Menus
        "Cross-Validation",
        "Diagnostic Information",
        "Time-Series",
        "Past Predictions"  # New Menu Item Added Here
    ])
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        # Save and upload the file
        file_id, webViewLink = upload_file_to_drive(uploaded_file.name, uploaded_file)
        
        # Store the link in a local file
        store_uploaded_file_link(uploaded_file.name, webViewLink)
        
        # Load data without headers
        df = pd.read_excel(uploaded_file, sheet_name='Converted 0 and 1', engine='openpyxl', header=None)

        # Read the cell value and strip whitespace
        cell_value = df.iloc[1, 0]
        if pd.isnull(cell_value):
            st.error("Cell A2 is empty. Please ensure 'establishment_name' is present in this cell.")
        elif str(cell_value).strip() != 'establishment_name':
            st.error(f"Expected 'establishment_name' in cell A2, but found '{cell_value}'. Please check your data.")
        else:
            # Proceed with processing
            # Extract categories from the first row
            categories = df.iloc[0]
            # Extract KPI names from the second row
            kpis = df.iloc[1]
            df = df.iloc[2:]  # Remove the category and KPI rows from the data
            df = df.reset_index(drop=True)  # Reset index after removing the first two rows

            # Set the column names to the KPI names
            df.columns = kpis.values

            # Set the indices of categories and kpis to match the DataFrame's columns
            categories.index = df.columns
            kpis.index = df.columns

            est_name_col = 'establishment_name'

            # Targets
            targets = ['satisfaction', 'value_for_money', 'return_probability', 'nps']

            # Exclude only 'establishment_name' and targets from categories and kpis, retain 'month' and 'Year'
            categories = categories.drop([est_name_col] + targets, errors='ignore')
            kpis = kpis.drop([est_name_col] + targets, errors='ignore')

            # Ensure that categories and kpis indices match the feature columns
            feature_columns = [col for col in df.columns if col not in [est_name_col, 'month', 'Year'] + targets]
            categories = categories[feature_columns]
            kpis = kpis[feature_columns]

            # Separate features and targets
            features = df[feature_columns]

            # Convert columns to numeric where possible
            features_converted = features.apply(pd.to_numeric, errors='coerce')

            # Keep only numeric columns
            features_converted = features_converted.select_dtypes(include=[np.number])

            # Check if features is empty
            if features_converted.empty:
                st.error("The features DataFrame is empty after selecting numeric columns. Please check your data.")
                st.stop()

            # Handle missing values using most frequent value
            imputer = SimpleImputer(strategy='most_frequent')
            X_imputed = pd.DataFrame(imputer.fit_transform(features_converted), columns=features_converted.columns)

            # Process targets (since data is already binary 0 and 1)
            y_imputed = df[targets].apply(pd.to_numeric, errors='coerce')
            y_imputed = y_imputed.fillna(y_imputed.mode().iloc[0])

            # Ensure targets are binary (0 and 1)
            for target in targets:
                y_imputed[target] = y_imputed[target].astype(int)

            # Standardize features
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X_imputed.columns)

            # Mapping for indicator names
            indicator_mapping = {
                'satisfaction': 'Wow Factor',
                'value_for_money': 'Perceived Value',
                'nps': 'NPS/Recommendation',
                'return_probability': 'Return Rate'
            }

            # Train models
            models, combined_scores = train_models(X_scaled, y_imputed, targets)

            # Handle different pages
            if page in ["All Stores: 7 KPIs", "Individual Store: 7 KPIs"]:
                if "Individual" in page:
                    # Individual Store: 7 KPIs
                    st.title("Individual Store: 7 KPIs Analysis")
                    selected_stores = select_stores(df, selector_key="individual_7_kpis_store_selector")
                    if selected_stores:
                        st.write(f"**Analysis for selected store(s):** {', '.join(selected_stores)}")
                        df_filtered = filter_dataframe(df, selected_stores)
                        # Update X_imputed, y_imputed, and X_scaled with filtered data
                        features_filtered = df_filtered[feature_columns]
                        features_converted_filtered = features_filtered.apply(pd.to_numeric, errors='coerce')
                        features_converted_filtered = features_converted_filtered.select_dtypes(include=[np.number])

                        if features_converted_filtered.empty:
                            st.error("After filtering, the features DataFrame is empty. Please check your data.")
                            st.stop()

                        X_imputed_filtered = pd.DataFrame(imputer.transform(features_converted_filtered), columns=features_converted_filtered.columns)
                        y_imputed_filtered = df_filtered[targets].apply(pd.to_numeric, errors='coerce')
                        y_imputed_filtered = y_imputed_filtered.fillna(y_imputed_filtered.mode().iloc[0])

                        for target in targets:
                            y_imputed_filtered[target] = y_imputed_filtered[target].astype(int)

                        X_scaled_filtered = pd.DataFrame(scaler.transform(X_imputed_filtered), columns=X_imputed_filtered.columns)

                        # Display Top 7 KPIs for the selected store(s)
                        display_top_7_kpis(
                            X_scaled_filtered, 
                            y_imputed_filtered, 
                            models, 
                            combined_scores, 
                            scaler, 
                            targets, 
                            indicator_mapping, 
                            kpis, 
                            X_imputed_filtered
                        )
                    else:
                        st.warning("Please select at least one store to proceed with the analysis.")
                else:
                    # All Stores: 7 KPIs
                    st.title("All Stores: 7 KPIs Analysis")
                    display_top_7_kpis(
                        X_scaled, 
                        y_imputed, 
                        models, 
                        combined_scores, 
                        scaler, 
                        targets, 
                        indicator_mapping, 
                        kpis, 
                        X_imputed
                    )

            elif page in ["All Stores: All KPIs", "Individual Store: All KPIs"]:
                if "Individual" in page:
                    # Individual Store: All KPIs
                    st.title("Individual Store: All KPIs Analysis")
                    selected_stores = select_stores(df, selector_key="individual_all_kpis_store_selector")
                    if selected_stores:
                        st.write(f"**Analysis for selected store(s):** {', '.join(selected_stores)}")
                        df_filtered = filter_dataframe(df, selected_stores)
                        # Update X_imputed, y_imputed, and X_scaled with filtered data
                        features_filtered = df_filtered[feature_columns]
                        features_converted_filtered = features_filtered.apply(pd.to_numeric, errors='coerce')
                        features_converted_filtered = features_converted_filtered.select_dtypes(include=[np.number])

                        if features_converted_filtered.empty:
                            st.error("After filtering, the features DataFrame is empty. Please check your data.")
                            st.stop()

                        X_imputed_filtered = pd.DataFrame(imputer.transform(features_converted_filtered), columns=features_converted_filtered.columns)
                        y_imputed_filtered = df_filtered[targets].apply(pd.to_numeric, errors='coerce')
                        y_imputed_filtered = y_imputed_filtered.fillna(y_imputed_filtered.mode().iloc[0])

                        for target in targets:
                            y_imputed_filtered[target] = y_imputed_filtered[target].astype(int)

                        X_scaled_filtered = pd.DataFrame(scaler.transform(X_imputed_filtered), columns=X_imputed_filtered.columns)

                        # Display All KPIs for the selected store(s)
                        display_all_kpis(
                            X_scaled_filtered, 
                            y_imputed_filtered, 
                            models, 
                            combined_scores, 
                            scaler, 
                            targets, 
                            indicator_mapping, 
                            kpis, 
                            X_imputed_filtered
                        )
                    else:
                        st.warning("Please select at least one store to proceed with the analysis.")
                else:
                    # All Stores: All KPIs
                    st.title("All Stores: All KPIs Analysis")
                    display_all_kpis(
                        X_scaled, 
                        y_imputed, 
                        models, 
                        combined_scores, 
                        scaler, 
                        targets, 
                        indicator_mapping, 
                        kpis, 
                        X_imputed
                    )

            elif page in ["All Stores: Top-Down Analysis", "Individual Store: Top-Down Analysis"]:
                st.title("Top-Down Analysis")
                if "Individual" in page:
                    selected_stores = select_stores(df, selector_key="top_down_store_selector")
                    if selected_stores:
                        st.write(f"**Analysis for selected store(s):** {', '.join(selected_stores)}")
                        df_filtered = filter_dataframe(df, selected_stores)
                        # Update X_imputed, y_imputed, and X_scaled with filtered data
                        features_filtered = df_filtered[feature_columns]
                        features_converted_filtered = features_filtered.apply(pd.to_numeric, errors='coerce')
                        features_converted_filtered = features_converted_filtered.select_dtypes(include=[np.number])

                        if features_converted_filtered.empty:
                            st.error("After filtering, the features DataFrame is empty. Please check your data.")
                            st.stop()

                        X_imputed_filtered = pd.DataFrame(imputer.transform(features_converted_filtered), columns=features_converted_filtered.columns)
                        y_imputed_filtered = df_filtered[targets].apply(pd.to_numeric, errors='coerce')
                        y_imputed_filtered = y_imputed_filtered.fillna(y_imputed_filtered.mode().iloc[0])

                        for target in targets:
                            y_imputed_filtered[target] = y_imputed_filtered[target].astype(int)

                        X_scaled_filtered = pd.DataFrame(scaler.transform(X_imputed_filtered), columns=X_imputed_filtered.columns)

                        # Perform Top-Down Analysis for the selected store(s)
                        top_down_analysis(
                            X_scaled_filtered, 
                            y_imputed_filtered, 
                            models, 
                            combined_scores, 
                            scaler, 
                            targets, 
                            indicator_mapping, 
                            X_imputed_filtered, 
                            button_key='individual_store_top_down'
                        )
                    else:
                        st.warning("Please select at least one store to proceed with the analysis.")
                else:
                    # All Stores: Top-Down Analysis
                    top_down_analysis(
                        X_scaled, 
                        y_imputed, 
                        models, 
                        combined_scores, 
                        scaler, 
                        targets, 
                        indicator_mapping, 
                        X_imputed, 
                        button_key='all_stores_top_down'
                    )

            elif page in ["All Stores: Correlation Matrix", "Individual Store: Correlation Matrix"]:
                if page == "All Stores: Correlation Matrix":
                    st.title("All Stores: Correlation Matrix")
                    # Prepare the numeric DataFrame similar to Diagnostic Information
                    correlation_numeric_df = X_imputed.copy()
                    display_correlation_matrix(correlation_numeric_df)
                else:
                    # Individual Store: Correlation Matrix
                    display_individual_correlation_matrix(df, X_imputed, display_correlation_matrix)

            elif page in ["All Stores: Category Impact", "Individual Store: Category Impact"]:
                if "Individual" in page:
                    # Individual Store: Category Impact
                    st.title("Individual Store: Category Impact")
                    selected_stores = select_stores(df, selector_key="category_impact_store_selector")
                    if selected_stores:
                        st.write(f"**Analysis for selected store(s):** {', '.join(selected_stores)}")
                        df_filtered = filter_dataframe(df, selected_stores)
                        # Update X_imputed, y_imputed, and X_scaled with filtered data
                        features_filtered = df_filtered[feature_columns]
                        features_converted_filtered = features_filtered.apply(pd.to_numeric, errors='coerce')
                        features_converted_filtered = features_converted_filtered.select_dtypes(include=[np.number])

                        if features_converted_filtered.empty:
                            st.error("After filtering, the features DataFrame is empty. Please check your data.")
                            st.stop()

                        X_imputed_filtered = pd.DataFrame(imputer.transform(features_converted_filtered), columns=features_converted_filtered.columns)
                        y_imputed_filtered = df_filtered[targets].apply(pd.to_numeric, errors='coerce')
                        y_imputed_filtered = y_imputed_filtered.fillna(y_imputed_filtered.mode().iloc[0])

                        for target in targets:
                            y_imputed_filtered[target] = y_imputed_filtered[target].astype(int)

                        X_scaled_filtered = pd.DataFrame(scaler.transform(X_imputed_filtered), columns=X_imputed_filtered.columns)

                        # Perform Category Impact Analysis for the selected store(s)
                        analyze_category_impact(
                            X_scaled_filtered, 
                            X_imputed_filtered, 
                            y_imputed_filtered, 
                            models, 
                            combined_scores, 
                            categories, 
                            kpis, 
                            scaler, 
                            targets, 
                            indicator_mapping
                        )
                    else:
                        st.warning("Please select at least one store to proceed with the analysis.")
                else:
                    # All Stores: Category Impact
                    st.title("All Stores: Category Impact")
                    analyze_category_impact(
                        X_scaled, 
                        X_imputed, 
                        y_imputed, 
                        models, 
                        combined_scores, 
                        categories, 
                        kpis, 
                        scaler, 
                        targets, 
                        indicator_mapping
                    )

            elif page == "Cross-Validation":
                st.title("Cross-Validation Results")
                for target in targets:
                    st.write(f"**{indicator_mapping[target]}:**")
                    
                    # Logistic Regression and Decision Tree
                    for model_name, model in models[target].items():
                        cv = 5  # Use 5-fold cross-validation
                        scores = cross_val_score(model, X_scaled, y_imputed[target], cv=cv, scoring='accuracy')
                        st.write(f"  - {model_name} Accuracy ({cv}-fold CV): {scores.mean():.2f}")
                    
                    # Combined CORREL-LINEST
                    combined_model = LinearRegression()
                    cv = 5  # Use 5-fold cross-validation
                    # Define a custom scorer that mimics accuracy
                    custom_scorer = make_scorer(lambda y_true, y_pred: accuracy_score(y_true, (y_pred > 0.5).astype(int)))
                    scores = cross_val_score(
                        combined_model, 
                        X_scaled, 
                        y_imputed[target], 
                        cv=cv, 
                        scoring=custom_scorer
                    )
                    st.write(f"  - Combined CORREL-LINEST Accuracy ({cv}-fold CV): {scores.mean():.2f}")
                    
                    st.write('---')
                
                # Removed Model Performance Comparison from Cross-Validation page

            elif page == "Diagnostic Information":
                st.title("Diagnostic Information")
                st.write("### Data Overview")
                st.write(f"**Total Rows:** {len(df)}")
                st.write(f"**Total Columns:** {len(df.columns)}")

                st.write("### First Few Rows of the Dataset")
                st.write(df.head())

                st.write("### Categories and KPIs")
                category_kpi_df = pd.DataFrame({
                    'Category': categories.values,
                    'KPI': kpis.values
                }, index=categories.index)
                st.write(category_kpi_df)

                st.write("### Column Information")
                column_info = pd.DataFrame({
                    'Column Name': df.columns,
                    'Data Type': df.dtypes,
                    'Non-Null Count': df.notnull().sum(),
                    'Null Count': df.isnull().sum(),
                    'Unique Values': df.nunique(),
                    'Category': categories.reindex(df.columns, fill_value='N/A').values
                })
                st.write(column_info)

                st.write("### Summary Statistics")
                # Convert columns to numeric where possible
                df_numeric_converted = df.apply(pd.to_numeric, errors='coerce')
                numeric_df = df_numeric_converted.select_dtypes(include=[np.number])
                if not numeric_df.empty and numeric_df.shape[1] > 0:
                    st.write(numeric_df.describe())
                else:
                    st.write("No numeric columns found in the dataset.")

                # Display Correlation Matrix
                st.write("### Correlation Matrix")
                display_correlation_matrix(numeric_df)
                
                # Add Model Performance Comparison
                display_model_performance_comparison(X_scaled, y_imputed, models, targets, indicator_mapping)

            elif page == "Time-Series":
                display_time_series_analysis(df.copy(), targets, indicator_mapping)

            elif page == "Past Predictions":
                st.title("Past Predictions")
                display_past_predictions()
    else:
        st.write("Please upload an Excel file to begin the analysis.")

if __name__ == "__main__":
    main()
