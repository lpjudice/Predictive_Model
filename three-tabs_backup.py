## backup em 9 de out 24
## modelo bottom up finalizado. Tentei alterar o sidebar menu algumas vezes, mas não funcionou. 
## Adicionei o debug para mostrar o exato impacto de cada KPI nas OKRs. 
## Já adicionado o Top Down. Decidi fazer o top down com todos os KPIs e nao apenas os 7, pois dava muita inconsistencia

## FALTA ADICIONAR A CORRELAÇÃO POR CATEGORIA (SERVICO/PRODUTO/ESTRUTURA - MEXER NA BASE)

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
import streamlit as st
from sklearn.impute import SimpleImputer
import warnings
import base64

# Custom CSS to style the app with the new sidebar design and improved metrics display
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

# Function to load and display the logo
def add_logo(logo_path):
    with open(logo_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    st.sidebar.markdown(
        f"""
        <img src="data:image/png;base64,{encoded_string}" class="logo-img">
        """,
        unsafe_allow_html=True
    )

# Helper functions
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
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1 Score': f1_score(y_true, y_pred, zero_division=0)
    }

# Function for top-down analysis
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
            current_percentage = (X_scaled[feature] > X_scaled[feature].mean()).mean() * 100
            
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

# Function to display and handle the "All KPIs" page
def display_all_kpis(X_scaled, y_imputed, models, combined_scores, scaler, targets, indicator_mapping):
    st.write("## All KPIs")
    
    with st.expander("Click here for explanations of performance metrics"):
        st.markdown("""
        <style>
        .small-font {
            font-size: 12px !important;
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
        current_percentage = (X_scaled[feature] > X_scaled[feature].mean()).mean() * 100
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
        X_modified = X_scaled.copy()
        any_changes = False
        debug_info = {}  # For storing debug information

        for feature in sorted_features.index:
            change = changes.get(feature, 0)
            if change != 0:
                any_changes = True
                feature_std = scaler.scale_[X_scaled.columns.get_loc(feature)]
                change_in_std_units = change / (100 * feature_std)
                X_modified[feature] += change_in_std_units
            
            # Store debug information
            debug_info[feature] = {
                'original': X_scaled[feature].mean(),
                'modified': X_modified[feature].mean(),
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

        for target in targets:
            st.markdown(f"### **{indicator_mapping[target]} Results:**")
            st.write("")  # Add space
            
            current_percentage = (y_imputed[target] == 1).mean() * 100
            
            for method in ['Logistic Regression', 'Decision Tree', 'Combined CORREL-LINEST']:
                st.markdown(f"**:blue[{method}:]**")
                
                if not any_changes:
                    st.write("No changes were made. The prediction remains the same as the current percentage.")
                    predicted_change = 0
                    final_predicted = current_percentage
                else:
                    if method in ['Logistic Regression', 'Decision Tree']:
                        model = models[target][method]
                        predictions = model.predict_proba(X_modified)[:, 1]
                    else:  # Combined CORREL-LINEST
                        combined_model = LinearRegression().fit(X_scaled, y_imputed[target])
                        predictions = combined_model.predict(X_modified)
                        predictions = np.clip(predictions, 0, 1)
                    
                    predicted_change = (np.mean(predictions) * 100) - current_percentage
                    final_predicted = current_percentage + predicted_change
                
                st.write(styled_metric("Current (%)", current_percentage))
                st.write(styled_metric("Predicted Change (Percentage Points)", predicted_change))
                st.write(styled_metric("Final Predicted (%)", final_predicted))
                
                st.write("---")  # Separator between methods

            st.write("---")  # Separator between indicators

# Main code
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

# Add logo to the sidebar
add_logo("logo_freatz.png")  # Replace with the actual path to your logo

if uploaded_file is not None:
    # Load data
    df = pd.read_excel(uploaded_file, sheet_name='Converted 0 and 1', engine='openpyxl')
    
    # Check dataset size and provide feedback
    dataset_size = len(df)
    if dataset_size < 100:
        st.warning(f"Your dataset is quite small ({dataset_size} samples). Results may not be statistically robust.")
        st.info("Consider expanding your dataset for more reliable results.")
    elif dataset_size < 1000:
        st.info(f"Your dataset is moderate in size ({dataset_size} samples). Results should be reasonably reliable, but more data could improve accuracy.")
    else:
        st.success(f"Your dataset is large ({dataset_size} samples). This should provide robust results.")
    
    # Separate features and targets
    targets = ['satisfaction', 'value_for_money', 'return_probability', 'nps']
    features = df.drop(columns=targets, errors='ignore')
    features = features.select_dtypes(include=[np.number])  # Keep only numeric columns

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = pd.DataFrame(imputer.fit_transform(features), columns=features.columns)
    y_imputed = df[targets].fillna(df[targets].mean()).round().astype(int)  # Ensure targets are integers

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
            
            if dataset_size < 100:
                logistic_model = LogisticRegression(max_iter=2000, C=1.0)  # Increase max_iter for small datasets
            else:
                logistic_model = LogisticRegression(max_iter=1000, C=1.0)
            
            logistic_model.fit(X_train, y_train)
            
            if any(isinstance(warn.message, ConvergenceWarning) for warn in w):
                st.warning("Logistic Regression may not have converged. Consider increasing max_iter.")
        
        if dataset_size < 100:
            tree_model = DecisionTreeClassifier(random_state=42, max_depth=3, min_samples_split=5)  # Reduced complexity for small datasets
        else:
            tree_model = DecisionTreeClassifier(random_state=42, max_depth=5, min_samples_split=10)
        
        tree_model.fit(X_train, y_train)
        
        models[target] = {'Logistic Regression': logistic_model, 'Decision Tree': tree_model}
        
        # Calculate combined score for this target
        combined_scores[target] = calculate_combined_score(X_scaled, y_imputed[target])

    # Updated sidebar menu
    st.sidebar.title("Predictive Model")
    
    st.sidebar.markdown('<div class="sidebar-menu">', unsafe_allow_html=True)
    page = st.sidebar.radio("", ["Top 7 KPIs", "All KPIs", "Cross-Validation", "Top Down Analysis"])
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

    if page == "Top 7 KPIs":
        # Model Performance Comparison
        st.write("## Model Performance Comparison")
        
        with st.expander("Click here for explanations of performance metrics"):
            st.markdown("""
            <style>
            .small-font {
                font-size: 12px !important;
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
            st.write(f"### {indicator_mapping[target]}")
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
                with st.expander(f"{i}. {feature}"):
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
                st.write(f"### {feature}")
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
                X_modified = X_scaled.copy()
                any_changes = False
                debug_info = {}  # For storing debug information

                for feature in top_7_features:
                    change = changes.get(feature, 0)
                    if change != 0:
                        any_changes = True
                        feature_std = scaler.scale_[X_scaled.columns.get_loc(feature)]
                        change_in_std_units = change / (100 * feature_std)
                        X_modified[feature] += change_in_std_units
                    
                    # Store debug information
                    debug_info[feature] = {
                        'original': X_scaled[feature].mean(),
                        'modified': X_modified[feature].mean(),
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

                for target in targets:
                    st.markdown(f"### **{indicator_mapping[target]} Results:**")
                    st.write("")  # Add space
                    
                    current_percentage = (y_imputed[target] == 1).mean() * 100
                    
                    for method in ['Logistic Regression', 'Decision Tree', 'Combined CORREL-LINEST']:
                        st.markdown(f"**:blue[{method}:]**")
                        
                        if not any_changes:
                            st.write("No changes were made. The prediction remains the same as the current percentage.")
                            predicted_change = 0
                            final_predicted = current_percentage
                        else:
                            if method in ['Logistic Regression', 'Decision Tree']:
                                model = models[target][method]
                                predictions = model.predict_proba(X_modified)[:, 1]
                            else:  # Combined CORREL-LINEST
                                combined_model = LinearRegression().fit(X_scaled, y_imputed[target])
                                predictions = combined_model.predict(X_modified)
                                predictions = np.clip(predictions, 0, 1)
                            
                            predicted_change = (np.mean(predictions) * 100) - current_percentage
                            final_predicted = current_percentage + predicted_change
                        
                        st.write(styled_metric("Current (%)", current_percentage))
                        st.write(styled_metric("Predicted Change (Percentage Points)", predicted_change))
                        st.write(styled_metric("Final Predicted (%)", final_predicted))
                        
                        st.write("---")  # Separator between methods

                    st.write("---")  # Separator between indicators

    elif page == "All KPIs":
        st.title("All KPIs Analysis")
        display_all_kpis(X_scaled, y_imputed, models, combined_scores, scaler, targets, indicator_mapping)

    elif page == "Cross-Validation":
        st.title("Cross-Validation Results")
        for target in targets:
            st.write(f"{indicator_mapping[target]}:")
            
            # Logistic Regression and Decision Tree
            for model_name, model in models[target].items():
                if dataset_size < 30:
                    cv = LeaveOneOut()
                    scores = cross_val_score(model, X_scaled, y_imputed[target], cv=cv, scoring='accuracy')
                    st.write(f'  {model_name} Accuracy (LOOCV): {scores.mean():.2f}')
                else:
                    cv = min(5, dataset_size // 10)  # Use 5-fold CV or less for smaller datasets
                    scores = cross_val_score(model, X_scaled, y_imputed[target], cv=cv, scoring='accuracy')
                    st.write(f'  {model_name} Accuracy ({cv}-fold CV): {scores.mean():.2f}')
            
            # Combined CORREL-LINEST
            combined_model = LinearRegression()
            if dataset_size < 30:
                cv = LeaveOneOut()
                scores = cross_val_score(combined_model, X_scaled, y_imputed[target], cv=cv, scoring=make_scorer(lambda y_true, y_pred: accuracy_score(y_true, (y_pred > 0.5).astype(int))))
                st.write(f'  Combined CORREL-LINEST Accuracy (LOOCV): {scores.mean():.2f}')
            else:
                cv = min(5, dataset_size // 10)  # Use 5-fold CV or less for smaller datasets
                scores = cross_val_score(combined_model, X_scaled, y_imputed[target], cv=cv, scoring=make_scorer(lambda y_true, y_pred: accuracy_score(y_true, (y_pred > 0.5).astype(int))))
                st.write(f'  Combined CORREL-LINEST Accuracy ({cv}-fold CV): {scores.mean():.2f}')
            
            st.write('---')

    elif page == "Top Down Analysis":
        st.title("Top Down Analysis")
        top_down_analysis(X_scaled, y_imputed, models, combined_scores, scaler, targets, indicator_mapping)

else:
    st.write("Please upload an Excel file to begin the analysis.")
