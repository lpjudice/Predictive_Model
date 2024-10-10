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
        font-size: 8px !important; /* Reduced font size by 4 points */
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
    color = "green" if (value > 0 and not reverse) or (value < 0 and reverse) else "red"
    formatted_value = f"{value:.2f}{'%' if is_percentage else ''}"
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

def select_stores(df):
    unique_stores = df['establishment_name'].unique()
    selected_stores = st.multiselect('Select store(s):', unique_stores)
    return selected_stores

def filter_dataframe(df, selected_stores):
    return df[df['establishment_name'].isin(selected_stores)]

# Analysis functions
def top_down_analysis(X_scaled, y_imputed, models, combined_scores, scaler, targets, indicator_mapping):
    st.write("## Top-Down Analysis")
    
    # Display current percentages and input desired goals
    st.write("### Current and Desired OKR Percentages:")
    current_percentages = {}
    desired_goals = {}
    col1, col2, col3 = st.columns(3)
    for idx, target in enumerate(targets):
        current_percentage = (y_imputed[target] == 1).mean() * 100
        current_percentages[target] = current_percentage
        with col1:
            st.write(f"{indicator_mapping[target]}")
        with col2:
            st.write(f"Current: {current_percentage:.2f}%")
        with col3:
            desired_goals[target] = st.number_input(f"Desired Goal for {indicator_mapping[target]} (%)", 
                                                    min_value=0.0, max_value=100.0, 
                                                    value=float(current_percentage), step=0.1, 
                                                    key=f"goal_{target}")
    
    # Select model
    model_choice = st.selectbox('Select Model for Analysis', 
                                ('Logistic Regression', 'Decision Tree', 'Combined CORREL-LINEST'))

    if st.button('Predict Required KPI Changes'):
        # Calculate required changes for each target
        required_changes = {}
        for target in targets:
            required_changes[target] = desired_goals[target] - current_percentages[target]
        
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
            current_percentage = (X_imputed[feature] > X_imputed[feature].mean()).mean() * 100
            
            # Calculate required change (this is a simplification and may need refinement)
            required_change = sum(required_changes.values()) * (importances[feature] / importances.sum())
            
            st.write(f"#### {feature}")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"Importance: {importances[feature]:.4f}")
            with col2:
                st.write(f"Current: {current_percentage:.2f}%")
            with col3:
                st.write(f"Required Change: {required_change:.2f} pp")
            st.write("---")

def display_all_kpis(X_scaled, y_imputed, models, combined_scores, scaler, targets, indicator_mapping):
    st.write("## All KPIs")
    
    with st.expander("Click here for explanations of performance metrics"):
        st.markdown("""
        <style>
        .small-font {
            font-size: 8px !important;
        }
        </style>
        <div class="small-font">
        <p><strong>Accuracy:</strong> The proportion of correct predictions (both true positives and true negatives) among the total number of cases examined.</p>
        
        <p><strong>Precision:</strong> The proportion of true positive predictions out of all positive predictions. It answers "Of all the instances the model predicted as positive, how many were actually positive?"</p>
        
        <p><strong>Recall:</strong> The proportion of true positive predictions out of all actual positive instances. It answers "Of all the actual positive instances, how many did the model correctly identify?"</p>
        
        <p><strong>F1 Score:</strong> The harmonic mean of precision and recall, providing a single score that balances both metrics.</p>
        
        <p><strong>Which metric to prioritize?</strong></p>
        <ul>
        <li>If missing positive cases is very costly, prioritize <strong>recall</strong>.</li>
        <li>If false positives are very costly, prioritize <strong>precision</strong>.</li>
        <li>If you need a balance between precision and recall, consider the <strong>F1 score</strong>.</li>
        <li>If overall correctness is most important, focus on <strong>accuracy</strong>.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    model_choice = st.selectbox('Select Model for KPI Ranking', 
                                ('Logistic Regression', 'Decision Tree', 'Combined CORREL-LINEST'))

    if model_choice == 'Logistic Regression':
        importances = pd.Series(np.abs(models['satisfaction']['Logistic Regression'].coef_[0]), index=X_scaled.columns)
    elif model_choice == 'Decision Tree':
        importances = pd.Series(models['satisfaction']['Decision Tree'].feature_importances_, index=X_scaled.columns)
    else:  # Combined CORREL-LINEST
        importances = combined_scores['satisfaction']

    sorted_features = importances.sort_values(ascending=False)

    st.write("## Enter desired changes:")
    changes = {}
    for feature in sorted_features.index:
        current_percentage = (X_imputed[feature] > X_imputed[feature].mean()).mean() * 100
        st.write(f"### {feature}")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"Importance: {importances[feature]:.4f}")
        with col2:
            st.write(f"Current: {current_percentage:.2f}%")
            changes[feature] = st.number_input(f"Change (pp)", value=0.0, step=0.1, format="%.1f", key=f"change_{feature}")
        with col3:
            new_percentage = current_percentage + changes[feature]
            new_gap = calculate_gap(new_percentage)
            st.write("After change:")
            st.markdown(color_metric("New %", new_percentage, reverse=True))
            st.markdown(color_metric("New Gap", new_gap))
        st.write("---")

    if st.button('Predict'):
        X_modified_imputed = X_imputed.copy()
        any_changes = False
        debug_info = {}  # For storing debug information

        for feature in sorted_features.index:
            change = changes.get(feature, 0)
            if change != 0:
                any_changes = True
                # Adjust the feature values to reach the desired mean
                current_mean = X_imputed[feature].mean()
                desired_mean = current_mean + (change / 100)
                desired_mean = min(max(desired_mean, 0), 1)  # Ensure it's between 0 and 1
                adjustment = desired_mean - current_mean
                X_modified_imputed[feature] += adjustment
                X_modified_imputed[feature] = X_modified_imputed[feature].clip(0, 1)

            # Store debug information
            debug_info[feature] = {
                'original': X_imputed[feature].mean(),
                'modified': X_modified_imputed[feature].mean(),
                'change': change
            }

        # Display debug information in an expander
        with st.expander("Click here to view debug information"):
            st.write("This information shows the changes applied to each feature:")
            for feature, info in debug_info.items():
                st.write(f"**{feature}**:")
                st.write(f"  Original mean: {info['original']:.4f}")
                st.write(f"  Modified mean: {info['modified']:.4f}")
                st.write(f"  User input change (pp): {info['change']:.1f}")
                st.write("---")

        if any_changes:
            X_modified_scaled = pd.DataFrame(scaler.transform(X_modified_imputed), columns=X_imputed.columns)
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

                    predicted_change = (np.mean(predictions) * 100) - current_percentage
                    final_predicted = current_percentage + predicted_change

                    st.write(styled_metric("Current (%)", current_percentage))
                    st.write(styled_metric("Predicted Change (Percentage Points)", predicted_change))
                    st.write(styled_metric("Final Predicted (%)", final_predicted))

                    st.write("---")  # Separator between methods

                st.write("---")  # Separator between indicators
        else:
            st.write("No changes were made. The prediction remains the same as the current percentage.")

def analyze_category_impact(X_scaled, X_imputed, y_imputed, models, combined_scores, categories, kpis, scaler, targets, indicator_mapping):
    st.write("## Category Impact Analysis")

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

    # Drop NaN categories
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

    # The rest of your existing function code goes here...
    # For brevity, I'll proceed to the category-based prediction

    # Category-based prediction
    st.write("## Category-based Prediction")
    changes = {}
    for category in category_importances_df.index:
        st.write(f"### {category}")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Category features:")
            category_features = categories[categories == category].index.tolist()
            category_features = [feat for feat in category_features if feat in X_imputed.columns]
            st.write(category_features)
            # For binary features (0/1), current percentage is mean * 100
            current_percentage = X_imputed[category_features].mean().mean() * 100
            st.write(f"Current: {current_percentage:.2f}%")
            changes[category] = st.number_input(f"Change for {category} (pp)", value=0.0, step=0.1, format="%.1f", key=f"change_{category}")
        with col2:
            new_percentage = current_percentage + changes[category]
            st.write("After change:")
            st.markdown(color_metric("New %", new_percentage, reverse=True))

    if st.button('Predict Category Impact'):
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
            for target in y_imputed.columns:
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
                    
                    predicted_change = (np.mean(predictions) * 100) - current_percentage
                    final_predicted = current_percentage + predicted_change
                    
                    st.write(styled_metric("Current (%)", current_percentage))
                    st.write(styled_metric("Predicted Change (Percentage Points)", predicted_change))
                    st.write(styled_metric("Final Predicted (%)", final_predicted))
                    
                    st.write("---")  # Separator between methods

                st.write("---")  # Separator between indicators
        else:
            st.write("No changes were made. The prediction remains the same as the current percentage.")

def display_top_7_kpis(X_scaled, X_imputed, y_imputed, models, combined_scores, scaler, targets, indicator_mapping, kpis):
    st.write("## Top 7 KPIs Analysis")
    
    # Model Performance Comparison
    st.write("### Model Performance Comparison")
    
    with st.expander("Click here for explanations of performance metrics"):
        st.markdown("""
        <style>
        .small-font {
            font-size: 8px !important;
        }
        </style>
        <div class="small-font">
        <p><strong>Accuracy:</strong> The proportion of correct predictions (both true positives and true negatives) among the total number of cases examined.</p>
        
        <p><strong>Precision:</strong> The proportion of true positive predictions out of all positive predictions. It answers "Of all the instances the model predicted as positive, how many were actually positive?"</p>
        
        <p><strong>Recall:</strong> The proportion of true positive predictions out of all actual positive instances. It answers "Of all the actual positive instances, how many did the model correctly identify?"</p>
        
        <p><strong>F1 Score:</strong> The harmonic mean of precision and recall, providing a single score that balances both metrics.</p>
        
        <p><strong>Which metric to prioritize?</strong></p>
        <ul>
        <li>If missing positive cases is very costly, prioritize <strong>recall</strong>.</li>
        <li>If false positives are very costly, prioritize <strong>precision</strong>.</li>
        <li>If you need a balance between precision and recall, consider the <strong>F1 score</strong>.</li>
        <li>If overall correctness is most important, focus on <strong>accuracy</strong>.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    accuracy_threshold = 0.65  # Set the accuracy threshold
    best_models = {}
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
        df_results['Meets_Accuracy_Threshold'] = df_results['Accuracy'] >= accuracy_threshold
        st.write(df_results.round(4))
        
        # Select best model
        valid_models = df_results[df_results['Meets_Accuracy_Threshold']]
        if not valid_models.empty:
            best_model = valid_models['Precision'].idxmax()
            st.write(f"Best model (highest Precision with Accuracy >= {accuracy_threshold}): **{best_model}**")
            best_models[target] = best_model
        else:
            st.write(f"No models meet the accuracy threshold of {accuracy_threshold}")
        
        st.write("---")

    st.write("Based on these metrics and the accuracy threshold, we've selected the best model for each target.")

    # Get feature importances (using satisfaction model)
    logistic_importances = pd.Series(np.abs(models['satisfaction']['Logistic Regression'].coef_[0]), index=X_scaled.columns)
    tree_importances = pd.Series(models['satisfaction']['Decision Tree'].feature_importances_, index=X_scaled.columns)
    combined_importances = combined_scores['satisfaction']

    # Select model for feature importance
    model_choice = st.selectbox('Select Model for Top 7 Features', 
                                ('Logistic Regression', 'Decision Tree', 'Combined CORREL-LINEST'))

    # Define top_7_features based on model choice
    if model_choice == 'Logistic Regression':
        top_7_features = logistic_importances.nlargest(7).index.tolist()
        importances = logistic_importances
    elif model_choice == 'Decision Tree':
        top_7_features = tree_importances.nlargest(7).index.tolist()
        importances = tree_importances
    else:  # Combined CORREL-LINEST
        top_7_features = combined_importances.nlargest(7).index.tolist()
        importances = combined_importances

    # Display top 7 features with gap calculation
    if top_7_features:
        st.write("## Top 7 Features:")
        for i, feature in enumerate(top_7_features, 1):
            current_percentage = (X_imputed[feature] > X_imputed[feature].mean()).mean() * 100
            gap = calculate_gap(current_percentage)
            
            # Create an expander for each feature
            with st.expander(f"{i}. {kpis[feature]}"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(color_metric("Importance", importances[feature], is_percentage=False))
                with col2:
                    st.markdown(color_metric("Current", current_percentage, reverse=True))
                with col3:
                    st.markdown(color_metric("Gap to 75%", gap))
                
                if model_choice == 'Combined CORREL-LINEST':
                    st.write("---")
                    st.write("Detailed Scores:")
                    correl = np.abs(pd.DataFrame(X_scaled).corrwith(y_imputed['satisfaction']))[feature]
                    linest = np.abs(LinearRegression().fit(X_scaled, y_imputed['satisfaction']).coef_[X_scaled.columns.get_loc(feature)])
                    st.write(f"CORREL: {correl:.4f}")
                    st.write(f"LINEST: {linest:.4f}")
                    st.write(f"Combined: {correl * linest:.4f}")
                
                # Progress bar for current percentage
                st.progress(min(current_percentage / 100, 1.0))
        st.write("---")
    else:
        st.write("Please select a model to view the Top 7 Features.")

    # Input desired percentage point change for each feature
    if top_7_features:
        st.write("## Enter desired changes:")
        changes = {}
        for feature in top_7_features:
            current_percentage = (X_imputed[feature] > X_imputed[feature].mean()).mean() * 100
            st.write(f"### {kpis[feature]}")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"Current: {current_percentage:.2f}%")
                changes[feature] = st.number_input(f"Change (pp)", value=0.0, step=0.1, format="%.1f", key=f"change_{feature}")
            with col2:
                new_percentage = current_percentage + changes[feature]
                new_gap = calculate_gap(new_percentage)
                st.write("After change:")
                st.markdown(color_metric("New %", new_percentage, reverse=True))
                st.markdown(color_metric("New Gap", new_gap))
            st.write("---")

        # Predict button
        if st.button('Predict'):
            X_modified_imputed = X_imputed.copy()
            any_changes = False
            debug_info = {}  # For storing debug information

            for feature in top_7_features:
                change = changes.get(feature, 0)
                if change != 0:
                    any_changes = True
                    # Adjust the feature values to reach the desired mean
                    current_mean = X_imputed[feature].mean()
                    desired_mean = current_mean + (change / 100)
                    desired_mean = min(max(desired_mean, 0), 1)  # Ensure it's between 0 and 1
                    adjustment = desired_mean - current_mean
                    X_modified_imputed[feature] += adjustment
                    X_modified_imputed[feature] = X_modified_imputed[feature].clip(0, 1)
                
                # Store debug information
                debug_info[feature] = {
                    'original': X_imputed[feature].mean(),
                    'modified': X_modified_imputed[feature].mean(),
                    'change': change
                }

            # Display debug information in an expander
            with st.expander("Click here to view debug information"):
                st.write("This information shows the changes applied to each feature:")
                for feature, info in debug_info.items():
                    st.write(f"**{kpis[feature]}**:")
                    st.write(f"  Original mean: {info['original']:.4f}")
                    st.write(f"  Modified mean: {info['modified']:.4f}")
                    st.write(f"  User input change (pp): {info['change']:.1f}")
                    st.write("---")

            if any_changes:
                X_modified_scaled = pd.DataFrame(scaler.transform(X_modified_imputed), columns=X_imputed.columns)
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

                        predicted_change = (np.mean(predictions) * 100) - current_percentage
                        final_predicted = current_percentage + predicted_change

                        st.write(styled_metric("Current (%)", current_percentage))
                        st.write(styled_metric("Predicted Change (Percentage Points)", predicted_change))
                        st.write(styled_metric("Final Predicted (%)", final_predicted))

                        st.write("---")  # Separator between methods

                    st.write("---")  # Separator between indicators
            else:
                st.write("No changes were made. The prediction remains the same as the current percentage.")

def display_diagnostic_info(df, categories, kpis, features, features_converted):
    st.title("Diagnostic Information")
    
    st.write("### Data Overview")
    st.write(f"Total number of rows: {len(df)}")
    st.write(f"Total number of columns: {len(df.columns)}")
    
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

    st.write("### Correlation Matrix")
    if not numeric_df.empty and numeric_df.shape[1] > 0:
        correlation_matrix = numeric_df.corr()
        # Apply background gradient
        styled_corr_matrix = correlation_matrix.style.background_gradient(cmap='Greens')
        st.dataframe(styled_corr_matrix)
    else:
        st.write("Correlation matrix cannot be computed because there are no numeric columns.")

    # Debug Information
    st.write("### Debug: Features DataFrame Before Processing")
    st.write(features.head())
    st.write("Features DataFrame Info Before Processing:")
    st.write(features.info())

    st.write("### Debug: Features DataFrame After Conversion to Numeric")
    st.write(features_converted.head())
    st.write("Features DataFrame Info After Conversion:")
    st.write(features_converted.info())

# Main code
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

# Add logo to the sidebar
add_logo("assets/logo_freatz.png")  # Adjust the path as needed

# Updated sidebar menu
st.sidebar.title("Navigation")

st.sidebar.markdown('<div class="sidebar-menu">', unsafe_allow_html=True)
page = st.sidebar.radio("", [
    "All Stores: Top 7 KPIs",
    "All Stores: All KPIs",
    "All Stores: Top-Down Analysis",
    "Individual Store: 7 KPIs",
    "Individual Store: All KPIs",
    "Individual Stores: Top-Down Analysis",
    "Category Impact: All Stores",
    "Category Impact: Individual Store",
    "Cross-Validation",
    "Diagnostic Information"
])
st.sidebar.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
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

        # Remove 'establishment_name' and targets from categories and kpis
        categories = categories.drop([est_name_col] + targets, errors='ignore')
        kpis = kpis.drop([est_name_col] + targets, errors='ignore')

        # Ensure that categories and kpis indices match the feature columns
        feature_columns = [col for col in df.columns if col not in [est_name_col] + targets]
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

        if "Individual Store" in page or page == "Category Impact: Individual Store":
            selected_stores = select_stores(df)
            if selected_stores:
                df_filtered = filter_dataframe(df, selected_stores)
                # Update X_imputed, y_imputed, and X_scaled with filtered data
                features = df_filtered[feature_columns]
                features_converted = features.apply(pd.to_numeric, errors='coerce')
                features_converted = features_converted.select_dtypes(include=[np.number])

                # Check if features is empty after filtering
                if features_converted.empty:
                    st.error("After filtering, the features DataFrame is empty. Please check your data.")
                    st.stop()

                X_imputed = pd.DataFrame(imputer.transform(features_converted), columns=features_converted.columns)
                y_imputed = df_filtered[targets].apply(pd.to_numeric, errors='coerce')
                y_imputed = y_imputed.fillna(y_imputed.mode().iloc[0])

                # Ensure targets are binary (0 and 1)
                for target in targets:
                    y_imputed[target] = y_imputed[target].astype(int)

                X_scaled = pd.DataFrame(scaler.transform(X_imputed), columns=X_imputed.columns)
            else:
                st.warning("Please select at least one store to proceed with the analysis.")
                st.stop()

        # Display current percentages for all targets with improved visuals
        st.write("## Current percentages for all indicators:")
        col1, col2 = st.columns(2)
        for idx, target in enumerate(targets):
            if target in y_imputed.columns:
                current_percentage = (y_imputed[target] == 1).mean() * 100
                with col1 if idx % 2 == 0 else col2:
                    st.metric(
                        label=indicator_mapping[target],
                        value=f"{current_percentage:.2f}%"
                    )
        st.write("---")

        # Train Logistic Regression and Decision Tree models for all targets
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

        if page == "All Stores: Top 7 KPIs" or page == "Individual Store: 7 KPIs":
            display_top_7_kpis(X_scaled, X_imputed, y_imputed, models, combined_scores, scaler, targets, indicator_mapping, kpis)
        
        elif page == "All Stores: All KPIs" or page == "Individual Store: All KPIs":
            st.title("All KPIs Analysis")
            display_all_kpis(X_scaled, y_imputed, models, combined_scores, scaler, targets, indicator_mapping)
        
        elif page == "All Stores: Top-Down Analysis" or page == "Individual Stores: Top-Down Analysis":
            st.title("Top Down Analysis")
            if "Individual" in page:
                st.write(f"Analysis for selected stores: {', '.join(selected_stores)}")
            top_down_analysis(X_scaled, y_imputed, models, combined_scores, scaler, targets, indicator_mapping)
        
        elif page == "Category Impact: All Stores" or page == "Category Impact: Individual Store":
            if "Individual" in page:
                st.write(f"Analysis for selected stores: {', '.join(selected_stores)}")
            analyze_category_impact(X_scaled, X_imputed, y_imputed, models, combined_scores, categories, kpis, scaler, targets, indicator_mapping)
        
        elif page == "Cross-Validation":
            st.title("Cross-Validation Results")
            for target in targets:
                st.write(f"{indicator_mapping[target]}:")
                
                # Logistic Regression and Decision Tree
                for model_name, model in models[target].items():
                    cv = 5  # Use 5-fold cross-validation
                    scores = cross_val_score(model, X_scaled, y_imputed[target], cv=cv, scoring='accuracy')
                    st.write(f'  {model_name} Accuracy ({cv}-fold CV): {scores.mean():.2f}')
                
                # Combined CORREL-LINEST
                combined_model = LinearRegression()
                cv = 5  # Use 5-fold cross-validation
                scores = cross_val_score(combined_model, X_scaled, y_imputed[target], cv=cv, scoring=make_scorer(lambda y_true, y_pred: accuracy_score(y_true, (y_pred > 0.5).astype(int))))
                st.write(f'  Combined CORREL-LINEST Accuracy ({cv}-fold CV): {scores.mean():.2f}')
                
                st.write('---')
        
        elif page == "Diagnostic Information":
            display_diagnostic_info(df, categories, kpis, features, features_converted)

else:
    st.write("Please upload an Excel file to begin the analysis.")
