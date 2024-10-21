from matplotlib.ticker import MultipleLocator
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

# New imports for Google Drive integration using OAuth 2.0
import os
import io
import csv
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from googleapiclient.errors import HttpError
import pickle
import json
import seaborn as sns
import textwrap
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# New imports for OpenAI Integration
import openai
from dotenv import load_dotenv  # To load environment variables from a .env file
from typing import List, Dict
import re
import traceback
from streamlit_option_menu import option_menu



# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI
def initialize_openai():
    """
    Initialize OpenAI API by setting the API key from environment variables.
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        st.error("OpenAI API key not found. Please set the 'OPENAI_API_KEY' environment variable.")
        st.stop()
    openai.api_key = openai_api_key

initialize_openai()



### COMMENTED THE CSS BECAUSE THE MONTSERRAT WAS TOO BIG FOR THE APP. I WASNT ABLE TO GET SMALLER SIZE, 12 WAS THE LIMIT

# # Custom CSS to style the app
# st.markdown("""
#     <style>
#     @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;700&display=swap');
    
#     html, body, [class*="css"], [class*="st-"], .stMarkdown, .stText, .stTable {
#         font-family: 'Montserrat', sans-serif !important;
#     }
    
#     .stButton>button {
#         background-color: #4CAF50;
#         color: white;
#         border-radius: 5px;
#         transition: background-color 0.3s ease;
#     }
    
#     .stButton>button:hover {
#         background-color: #45a049;
#     }
    
#     .stSelectbox [data-baseweb="select"] {
#         background-color: white;
#         border-radius: 5px;
#     }
    
#     .stExpander {
#         background-color: white;
#         border-radius: 5px;
#         border: 1px solid #e0e0e0;
#     }
    
#     .sidebar .sidebar-content {
#         background-color: #f0f2f6;
#     }
    
#     .sidebar .sidebar-content .sidebar-menu {
#         border-top: 1px solid #e0e0e0;
#         border-bottom: 1px solid #e0e0e0;
#         padding: 1rem 0;
#         margin: 1rem 0;
#     }
    
#     .sidebar .sidebar-content .sidebar-menu .stRadio > div {
#         padding: 0.5rem 0;
#         cursor: pointer;
#         transition: background-color 0.3s ease;
#         font-size: 12px !important;
#     }
    
#     .sidebar .sidebar-content .sidebar-menu .stRadio > div:hover {
#         background-color: #e0e0e0;
#     }
    
#     .sidebar .sidebar-content .sidebar-menu .stRadio > div[data-checked="true"] {
#         font-weight: bold;
#     }
    
#     .logo-img {
#         display: block;
#         margin: 20px auto;
#         width: 100px;
#         position: sticky;
#         top: 0;
#         z-index: 999;
#     }
#     </style>
#     """, unsafe_allow_html=True)

# Define columns to exclude from analysis
base_exclude_columns = ['establishment_name', 'month', 'Year', 'Date', "nps2"]
comment_columns = ['Comments 1', 'Comments 2', 'Comments 3', 'Comments 4', 'Comments 5']
exclude_columns = base_exclude_columns + comment_columns

# Define target variables
targets = ['satisfaction', 'value_for_money', 'return_probability', 'nps']

def calculate_gap_to_75_percent(value):
    if isinstance(value, pd.Series):
        return np.maximum(75 - value, 0).iloc[0]  # Return the first value if it's a Series
    else:
        return max(75 - value, 0)

def extract_matches(text: str) -> List[Dict]:
    matches = []
    lines = text.split('\n')
    current_match = {}
    
    for line in lines:
        line = line.strip()
        if line.startswith('{') and line.endswith('}'):
            try:
                match = eval(line)
                if isinstance(match, dict) and 'Grouped_Metrics' in match and 'similarity_score' in match:
                    matches.append(match)
            except:
                pass
        elif 'Grouped_Metrics:' in line:
            if current_match:
                matches.append(current_match)
                current_match = {}
            current_match['Grouped_Metrics'] = line.split('Grouped_Metrics:')[1].strip()
        elif 'Similarity Score:' in line:
            score = line.split('Similarity Score:')[1].strip()
            if score.endswith('%'):
                current_match['similarity_score'] = float(score.rstrip('%')) / 100
            elif score.lower() == 'high':
                current_match['similarity_score'] = 0.9
            elif score.lower() == 'medium':
                current_match['similarity_score'] = 0.6
            elif score.lower() == 'low':
                current_match['similarity_score'] = 0.3
            else:
                try:
                    current_match['similarity_score'] = float(score)
                except ValueError:
                    current_match['similarity_score'] = 0.5
            matches.append(current_match)
            current_match = {}
    
    if current_match:
        matches.append(current_match)
    
    return matches


def get_action_items(grouped_metric: str, kpi_actions: pd.DataFrame) -> List[str]:
    # Find the row in kpi_actions that matches the grouped_metric
    matching_row = kpi_actions[kpi_actions['Grouped_Metrics'] == grouped_metric]
    
    if matching_row.empty:
        return []  # Return an empty list if no match is found
    
    # Get the Action_Items from the matching row
    action_items = matching_row['Action_Items'].iloc[0]
    
    # Check if action_items is a string (it should be)
    if isinstance(action_items, str):
        # Split the string into a list, assuming items are separated by semicolons
        actions_list = [action.strip() for action in action_items.split(';') if action.strip()]
        return actions_list[:10]  # Return up to 10 action items
    else:
        return []  # Return an empty list if Action_Items is not a string

def get_openai_matches(metric: str, grouped_metrics: List[Dict], n: int = 5) -> List[Dict]:
    try:
        # Prepare the prompt for OpenAI
        prompt = f"Match the following metric to the most relevant Grouped_Metrics: {metric}\n\n"
        prompt += "Grouped_Metrics:\n"
        for gm in grouped_metrics:
            prompt += f"- {gm['Grouped_Metrics']}: {gm['Why_Improving_It']} (Importance: {gm['KPI_Importance']})\n"
        prompt += f"\nReturn the top {n} matches as a list, each item containing 'Grouped_Metrics' and 'similarity_score'."

        # Make API call to OpenAI
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that matches metrics to Grouped_Metrics."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
        )

         # Get the response content
        response_content = response.choices[0].message['content'].strip()
        print(f"Raw OpenAI response: {response_content}")  # Debug print

        # Use the improved extract_matches function
        matches = extract_matches(response_content)

        print(f"Parsed matches: {matches}")  # Debug print

        # Ensure we have the correct number of matches
        matches = matches[:n]

        return matches
    except Exception as e:
        print(f"Error in get_openai_matches: {str(e)}")
        print(traceback.format_exc())
        return []

# Helper functions
def add_logo(logo_path):
    try:
        with open(logo_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        st.sidebar.markdown(
            f"""
            <img src="data:image/png;base64,{encoded_string}" class="logo-img" style="width: 50%; height: auto; display: block; margin: 0 auto;">
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
        return f"{label}: **{value}**"
    
    return f"{label}: **:{color}[{formatted_value}]**"

# Helper function to find the top 5 best matches for a KPI from a list of Short KPIs
def find_top_matches_by_context(kpi_importance, short_kpis_importance, top_n=5):
    """
    Uses contextual similarity (TF-IDF) to match KPIs based on context rather than direct text similarity.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    
    # Fit the TF-IDF vectorizer to the context
    tfidf_vectorizer = TfidfVectorizer().fit_transform([kpi_importance] + short_kpis_importance)
    similarity_matrix = cosine_similarity(tfidf_vectorizer[0:1], tfidf_vectorizer[1:])
    
    top_match_indices = np.argsort(similarity_matrix[0])[-top_n:][::-1]  # Get top N matches
    
    return [short_kpis_importance[idx] for idx in top_match_indices], similarity_matrix[0][top_match_indices]



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


def get_action_plan(store_name, all_comments):
    """
    Generate the action plan for a given store using OpenAI.
    """
    prompt = (
        f"Store name: {store_name}: Area Focus in short terms\n"
        f"Issue:\n"
        f"Recommendations:\n\n"
        f"Based on the comments provided below (could be any language, but your answer has to be in Enghlish), and assuming the role of an experienced business analyst, "
        f"identify the most frequent complaints or strenghts among customers that lead to them becoming highly Satisfied or Detractors. "
        f"Please categorize the weakeness or strength by theme, and pick one to be the focus of the month. You should provide the percentage of times each complaint is mentioned, "
        f"and explain why this was the choice amongst all other potential issues, highlighting any patterns or insights that may explain why these issues are driving the picked focuse.\n\n"
        f"Comments:\n{all_comments}"
    )
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7,
        )
        answer = response['choices'][0]['message']['content']
        return answer
    except Exception as e:
        return f"An error occurred while communicating with OpenAI: {e}"


# Google Drive Integration Functions

def creds_to_dict(creds):
    return {
        'token': creds.token,
        'refresh_token': creds.refresh_token,
        'token_uri': creds.token_uri,
        'client_id': creds.client_id,
        'client_secret': creds.client_secret,
        'scopes': creds.scopes
    }

def authenticate_google_drive():
    SCOPES = ['https://www.googleapis.com/auth/drive.file']

    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    else:
        flow = InstalledAppFlow.from_client_secrets_file('desktopapp-v3.json', SCOPES)
        creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return creds

def get_folder_id(service, folder_name):
    try:
        query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
        results = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
        items = results.get('files', [])

        if items:
            return items[0]['id']
        else:
            file_metadata = {
                'name': folder_name,
                'mimeType': 'application/vnd.google-apps.folder'
            }
            folder = service.files().create(body=file_metadata, fields='id').execute()
            return folder.get('id')
    except HttpError as error:
        st.error(f"An error occurred while accessing Google Drive: {error}")
        return None

def upload_file_to_drive(file_name, file_data):
    creds = authenticate_google_drive()
    service = build('drive', 'v3', credentials=creds)

    folder_name = "Past Predictions - Freatz Predictive Model"
    folder_id = get_folder_id(service, folder_name)

    if folder_id:
        try:
            file_metadata = {
                'name': file_name,
                'parents': [folder_id]
            }
            fh = io.BytesIO(file_data.getvalue())
            media = MediaIoBaseUpload(
                fh, 
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
            uploaded_file = service.files().create(
                body=file_metadata, 
                media_body=media, 
                fields='id, webViewLink'
            ).execute()

            return uploaded_file.get('id'), uploaded_file.get('webViewLink')
        except HttpError as error:
            st.error(f"An error occurred while uploading the file: {error}")
            return None, None
    else:
        st.error(f"Unable to locate or create the folder '{folder_name}'. Please ensure the folder exists or check your Google Drive permissions.")
        return None, None

def store_uploaded_file_link(file_name, link):
    if link is None:
        st.error("Failed to upload the file. No link available to store.")
        return

    file_exists = os.path.isfile('uploaded_files.csv')
    with open('uploaded_files.csv', 'a', newline='') as csvfile:
        fieldnames = ['timestamp', 'file_name', 'link']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({'timestamp': datetime.now(), 'file_name': file_name, 'link': link})

def display_past_predictions():
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

def display_plan_of_action(df, X_imputed, y_imputed, models, combined_scores, scaler, targets, indicator_mapping, kpi_actions_upper, kpi_actions_consumers, kpi_actions_freatz, column_metadata):
    st.header("All Stores: Plan of Action")
    
    # Add this code to display the warning message**
    st.markdown(
        """
        <div style="background-color:#f8d7da; padding: 10px; border-radius: 5px; font-size: 12px;">
            <strong>Note:</strong> This page may take some time to load, depending on the number of stores. The content is AI-generated in real time, drawing from thousands of data points derived from consumer comments about each store.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    # --- Brand-Wide Plan of Action ---
    with st.expander("Brand: Overall Focus"):
        # Define comments columns
        comments_columns = ['Comments 1', 'Comments 2', 'Comments 3', 'Comments 4', 'Comments 5']
        comments_columns = [col for col in comments_columns if col in df.columns]
        
        if not comments_columns:
            st.write("No comments available for the brand.")
        else:
            # Concatenate all comments across all stores
            comments_series = df[comments_columns].astype(str).agg(' '.join, axis=1)
            all_comments = ' '.join(comments_series.tolist())
            
            # Generate action plan using cached function
            action_plan = get_action_plan("Brand", all_comments)
            
            # Display the action plan
            st.markdown(action_plan)
    
    st.markdown("---")


    # --- Individual Store Plan of Action ---

    # Get unique stores sorted alphabetically
    stores = sorted(df['establishment_name'].dropna().unique())
    
    for store in stores:
        with st.expander(f"{store}: Main Focus"):
            # Filter comments for the store
            store_comments = df[df['establishment_name'] == store]
            comments_columns = ['Comments 1', 'Comments 2', 'Comments 3', 'Comments 4', 'Comments 5']
            comments_columns = [col for col in comments_columns if col in store_comments.columns]
            
            if not comments_columns:
                st.write("No comments available for this store.")
                continue
            
            # Concatenate comments
            comments_series = store_comments[comments_columns].astype(str).agg(' '.join, axis=1)
            all_comments = ' '.join(comments_series.tolist())
            
            # Generate action plan using cached function
            action_plan = get_action_plan(store, all_comments)
            
            # Display the action plan
            st.markdown(action_plan)


    # 1. Target Metrics Summary
    st.subheader("Target Metrics Summary")
    metrics_cols = st.columns(len(targets))
    for idx, target in enumerate(targets):
        current_percentage = (y_imputed[target] == 1).mean() * 100
        metrics_cols[idx].metric(
            label=indicator_mapping[target],
            value=f"{current_percentage:.2f}%",
            delta=None
        )
    st.markdown("---")
    
    # 2. Model Selection Dropdown
    st.subheader("Actions Items to Improve Each KPI")
    selected_model = st.selectbox(
        "Choose a model to base the action plan on:",
        ("Logistic Regression", "Decision Tree", "Combined CORREL-LINEST"),
        key="plan_of_action_model_selection"
    )
    
    # 3. Load KPIs Based on Selected Model
    st.subheader("KPIs Overview")
    if selected_model == "Logistic Regression":
        kpi_importances = pd.Series(np.abs(models['satisfaction']['Logistic Regression'].coef_[0]), index=X_imputed.columns)
    elif selected_model == "Decision Tree":
        kpi_importances = pd.Series(models['satisfaction']['Decision Tree'].feature_importances_, index=X_imputed.columns)
    else:
        kpi_importances = combined_scores['satisfaction']
    
    sorted_kpis = kpi_importances.sort_values(ascending=False).index.tolist()
    
    # 4. Display KPIs with Enhanced Visuals
    for kpi in sorted_kpis:
        kpi_info = column_metadata[column_metadata['Column Name'] == kpi].iloc[0]
        category = kpi_info['Category']
        moment = kpi_info['Moment']
        gap = calculate_gap_to_75_percent(X_imputed[kpi].mean() * 100)
        current_rating = X_imputed[kpi].mean() * 100  # Assuming rating is the mean percentage
        
        # 4.1. Display Category | Moment
        st.markdown(f"**<span style='font-size:12px;'>{category} | {moment}</span>**", unsafe_allow_html=True)
        
        # 4.2. Display KPI with color-coded rating and gap
        color = 'green' if current_rating >= 75 else 'orange' if current_rating >= 60 else 'red'
        kpi_display = f"<span style='font-size:14px; color:{color};'>• {kpi}: {current_rating:.1f}% (Gap: {gap:.1f}%)</span>"
        st.markdown(kpi_display, unsafe_allow_html=True)
        
        # 4.3. Prepare Grouped_Metrics data and get OpenAI matches
        all_grouped_metrics = (
            kpi_actions_upper[['Grouped_Metrics', 'Why_Improving_It', 'KPI_Importance']].to_dict('records') +
            kpi_actions_consumers[['Grouped_Metrics', 'Why_Improving_It', 'KPI_Importance']].to_dict('records') +
            kpi_actions_freatz[['Grouped_Metrics', 'Why_Improving_It', 'KPI_Importance']].to_dict('records')
        )
        matches = get_openai_matches(kpi, all_grouped_metrics)

        # New: Check if matches were found
        if not matches:
            st.warning(f"No matches found for {kpi}. Please try again or refine your search.")
            continue

        # 4.4. Display expanders for each source with action items
        for source, kpi_actions in [("Upper Management", kpi_actions_upper), 
                                ("Consumers", kpi_actions_consumers), 
                                ("Freatz", kpi_actions_freatz)]:
            with st.expander(f"**Actions Suggested by {source}**"):
                source_matches = [m for m in matches if m['Grouped_Metrics'] in kpi_actions['Grouped_Metrics'].values]
                if source_matches:
                    for match in source_matches:
                        grouped_metric = match['Grouped_Metrics']
                        similarity_score = match['similarity_score']
                        st.markdown(f"**{grouped_metric}** (Similarity: {similarity_score:.2f})")
                
                        actions = get_action_items(grouped_metric, kpi_actions)
                        if actions:
                            for idx, action in enumerate(actions, 1):
                                st.markdown(f"{idx}. {action.strip()}")
                        else:
                            st.warning(f"No action items found for {grouped_metric}")
                
                        st.markdown("---")
                else:
                    st.info(f"No matching Grouped_Metrics found for {source}")

    st.markdown("---")

def get_kpi_importance(kpi, kpi_actions):
    """
    Get the 'KPI_Importance' for a given KPI from the relevant KPIs_to_Actions sheet.
    """
    kpi_importance = kpi_actions.loc[kpi_actions['Short KPI'] == kpi, 'KPI_Importance']
    if not kpi_importance.empty:
        return kpi_importance.values[0]
    else:
        return None  # Handle cases where no KPI_Importance is found
    
def display_expander(title, matches, kpi_actions):
    """
    Displays an expander for a specific source (Freatz, Consumers, Upper Management) with matching Short KPIs.
    """
    with st.expander(f"**{title}**"):
        if matches:
            for idx, match in enumerate(matches):
                short_kpi = match['Short KPI']
                actions = match['Action_Items']
                why_improving = match['Why_Improving_It']

                st.markdown(f"**{idx+1}. {short_kpi}**")
                st.markdown(f"<i>{actions}</i>", unsafe_allow_html=True)
                
                # Tooltip implementation for Why_Improving_It
                st.markdown(f"<span style='color:gray; font-size:11px;'>{why_improving}</span>", unsafe_allow_html=True)
                
                st.markdown("---")  # Separator between Short KPIs
        else:
            st.write("No matching Short KPIs found.")

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
        
        combined_scores[target] = calculate_combined_score(X_scaled, y_imputed[target])
    
    return models, combined_scores

# Model Performance Comparison
def display_model_performance_comparison(X_scaled, y_imputed, models, targets, indicator_mapping):
    with st.expander("Model Performance Comparison"):
        st.write("### Model Performance Comparison")
        
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
            combined_model = LinearRegression().fit(X_scaled, y_imputed[target])
            combined_pred = (combined_model.predict(X_test) > 0.5).astype(int)
            results['Combined CORREL-LINEST'] = calculate_metrics(y_test, combined_pred)
            
            # Display results
            df_results = pd.DataFrame(results).T
            st.write(df_results.round(4))
            
            # Determine the best model
            accuracies = df_results['Accuracy']
            eligible_models = accuracies[accuracies >= 0.65].index

            if not eligible_models.empty:
                precisions = df_results.loc[eligible_models, 'Precision']
                best_model = precisions.idxmax()
                st.write(f"Best model (highest Precision with Accuracy >= 0.65): **{best_model}**")
            else:
                st.write("No model meets the criteria of Accuracy >= 0.65.")
            
            st.write("---")

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
            
            required_change = sum(required_changes.values()) * (importances[feature] / importances.sum())
            
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
    
    # Select model
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
    
    # Select model
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
    
    # Select model
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

    # Calculate per feature importances
    corr = np.abs(pd.DataFrame(X_scaled).corrwith(y_imputed['satisfaction']))
    corr.index = X_scaled.columns

    linreg_model = LinearRegression().fit(X_scaled, y_imputed['satisfaction'])
    linreg_coef = pd.Series(np.abs(linreg_model.coef_), index=X_scaled.columns)

    combined_score = corr * linreg_coef

    logistic_model = models['satisfaction']['Logistic Regression']
    logistic_coef = pd.Series(np.abs(logistic_model.coef_[0]), index=X_scaled.columns)

    tree_model = models['satisfaction']['Decision Tree']
    tree_importance = pd.Series(tree_model.feature_importances_, index=X_scaled.columns)

    feature_importances_df = pd.DataFrame({
        'Correlation': corr,
        'LinReg_Coefficient': linreg_coef,
        'Combined_Score': combined_score,
        'Logistic_Coefficient': logistic_coef,
        'DecisionTree_Importance': tree_importance
    })

    feature_importances_df['Category'] = categories.reindex(X_scaled.columns)
    
    feature_importances_df = feature_importances_df.dropna(subset=['Category'])

    category_importances_df = feature_importances_df.groupby('Category').sum()

    category_importances_df = category_importances_df.rename(columns={
        'LinReg_Coefficient': 'Impact',
        'Combined_Score': 'Combined CORREL-LINEST',
        'Logistic_Coefficient': 'Logistic Regression Importance',
        'DecisionTree_Importance': 'Decision Tree Importance'
    })

    category_importances_df = category_importances_df[['Correlation', 'Impact', 'Combined CORREL-LINEST',
                                                      'Logistic Regression Importance', 'Decision Tree Importance']]

    category_importances_df = category_importances_df.round(4)

    category_importances_df = category_importances_df.sort_values('Combined CORREL-LINEST', ascending=False)

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
                for feat in category_features:
                    current_mean = X_imputed[feat].mean()
                    desired_mean = current_mean + (change / 100)
                    desired_mean = min(max(desired_mean, 0), 1)  # Ensure it's between 0 and 1
                    adjustment = desired_mean - current_mean
                    X_modified_imputed[feat] += adjustment
                    X_modified_imputed[feat] = X_modified_imputed[feat].clip(0, 1)

        if any_changes:
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

                    final_predicted = min(max(final_predicted, 0.0), 100.0)
                    predicted_change = final_predicted - current_percentage

                    st.metric("Current (%)", f"{current_percentage:.2f}%", delta=None)
                    st.metric("Predicted Change (pp)", f"{predicted_change:.2f} pp", delta=predicted_change)
                    st.metric("Final Predicted (%)", f"{final_predicted:.2f}%", delta=predicted_change)
                    
                    st.write("---")  # Separator between methods

                st.write("---")  # Separator between indicators
        else:
            st.info("No changes were made. The prediction remains the same as the current percentage.")

def display_individual_correlation_matrix(df, X_imputed, display_correlation_matrix_func):
    st.title("Individual Store: Correlation Matrix")
    
    selected_stores = select_stores(df, selector_key="individual_correlation_matrix_store_selector")
    
    if selected_stores:
        df_filtered = filter_dataframe(df, selected_stores)
        feature_columns = [col for col in df.columns if col not in exclude_columns + targets]
        features_filtered = df_filtered[feature_columns]
        
        features_converted = features_filtered.apply(pd.to_numeric, errors='coerce')
        features_converted = features_converted.select_dtypes(include=[np.number])
        
        if features_converted.empty:
            st.error("No numeric features available for the selected store(s).")
            return
        
        imputer = SimpleImputer(strategy='most_frequent')
        X_filtered_imputed = pd.DataFrame(imputer.fit_transform(features_converted), columns=features_converted.columns)
        
        st.write("### Correlation Matrix for Selected Store(s)")
        display_correlation_matrix_func(df, X_filtered_imputed)
    else:
        st.warning("Please select at least one store to view its correlation matrix.")

def display_correlation_matrix(df, numeric_df):
    st.write("### Correlation Matrix")
    if not numeric_df.empty and numeric_df.shape[1] > 0:
        target_columns = ['satisfaction', 'value_for_money', 'return_probability', 'nps']
        
        numeric_df = pd.concat([numeric_df, df[target_columns]], axis=1, join='inner')
        
        correlation_matrix = numeric_df.drop(columns=exclude_columns, errors='ignore').corr()

        styled_corr_matrix = correlation_matrix.style.background_gradient(cmap='Greens')
        st.dataframe(styled_corr_matrix)
    else:
        st.write("Correlation matrix cannot be computed because there are no numeric columns.")

def display_time_series_analysis(df, targets, indicator_mapping):
    st.title("Time-Series Analysis")

    if 'month' not in df.columns or 'Year' not in df.columns:
        st.error("The dataset must contain 'month' and 'Year' columns for Time-Series analysis.")
        return

    try:
        df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['month'].astype(str), format='%Y-%m')
    except Exception as e:
        st.error(f"Error parsing 'month' and 'Year' columns into datetime: {e}")
        return

    df = df.sort_values('Date')

    for target in targets:
        if target not in df.columns:
            st.error(f"Target variable '{target}' not found in the dataset.")
            return

    st.sidebar.header("Time-Series Filters")
    min_date = df['Date'].min()
    max_date = df['Date'].max()

    start_date = st.sidebar.date_input("Start date", min_date, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("End date", max_date, min_value=min_date, max_value=max_date)

    if start_date > end_date:
        st.error("Start date must be before or equal to End date.")
        return

    st.sidebar.header("Forecasting Model Selection")
    model_choice = st.sidebar.selectbox(
        "Select the forecasting model:",
        ("Linear Regression", "Exponential Smoothing")
    )

    df_filtered = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))].copy()
    

    if df_filtered.empty:
        st.warning("No data available for the selected date range.")
        return

    for target in targets:
        df_filtered[target] = pd.to_numeric(df_filtered[target], errors='coerce')

    df_filtered[targets] = df_filtered[targets].fillna(0)

    df_grouped = df_filtered.groupby('Date').agg({target: 'mean' for target in targets})
    df_grouped = df_grouped * 100  # Convert to percentage

    st.write(f"### Data from {start_date} to {end_date}")
    st.dataframe(df_grouped.style.format("{:.2f}%"))

    st.write("### Trend Projection for Next 3 Months")

    projections = {}

    for target in targets:
        if df_grouped.shape[0] < 3:
            st.warning(f"Not enough data points to perform trend analysis for {indicator_mapping[target]}.")
            continue

        y = df_grouped[target]

        y = y.fillna(method='ffill').fillna(method='bfill')

        if model_choice == "Linear Regression":
            df_trend = df_grouped.copy()
            df_trend.reset_index(inplace=True)
            df_trend['Month_Num'] = df_trend['Date'].dt.year * 12 + df_trend['Date'].dt.month

            X = df_trend[['Month_Num']]
            y_values = df_trend[target]

            model = LinearRegression()
            model.fit(X, y_values)

            last_month_num = df_trend['Month_Num'].iloc[-1]
            future_month_nums = [last_month_num + i for i in range(1, 4)]
            future_dates = [df_trend['Date'].iloc[-1] + pd.DateOffset(months=i) for i in range(1, 4)]
            predicted_values = model.predict(np.array(future_month_nums).reshape(-1, 1))

            predicted_values = np.array(predicted_values)

            model_description = "Linear Regression"

        elif model_choice == "Exponential Smoothing":
            model = ExponentialSmoothing(y, trend='add', seasonal=None, initialization_method="estimated")
            model_fit = model.fit()

            predicted_values = model_fit.forecast(steps=3)

            predicted_values = np.array(predicted_values)

            future_dates = [df_grouped.index[-1] + pd.DateOffset(months=i) for i in range(1, 4)]

            model_description = "Exponential Smoothing"

        else:
            st.error("Selected model is not implemented.")
            return

        current_value = y.iloc[-3:].mean()

        predicted_value = predicted_values[-1]
        change_pp = predicted_value - current_value

        if change_pp > 0:
            trend_direction = 'Increasing'
            arrow_color = 'green'
        elif change_pp < 0:
            trend_direction = 'Decreasing'
            arrow_color = 'red'
        else:
            trend_direction = 'Stable'
            arrow_color = 'black'

        predicted_value = min(max(predicted_value, 0.0), 100.0)

        projections[target] = {
            'Current 3-Month Average (%)': current_value,
            'Predicted Value in 3 Months (%)': predicted_value,
            'Change (pp)': change_pp,
            'Trend': trend_direction
        }

        st.write(f"#### {indicator_mapping[target]} Projection")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df_grouped.index, y, marker='o', linestyle='-', label='Actual')
        ax.set_xlabel('Date')
        ax.set_ylabel(f"{indicator_mapping[target]} (%)")
        ax.set_title(f"{indicator_mapping[target]} Over Time with Projection")
        ax.grid(True)

        average_date = df_grouped.index[-1]
        ax.scatter(average_date, current_value, color='red', zorder=5, label='3-Month Avg')

        ax.annotate(f"{current_value:.2f}%", (average_date, current_value),
                    textcoords="offset points", xytext=(0,10), ha='center', va='bottom', color='red')

        future_date = future_dates[-1]
        predicted_value_capped = min(max(predicted_value, 0), 100)
        ax.scatter(future_date, predicted_value_capped, color='orange', zorder=5, label='Predicted Value')

        ax.annotate(f"{predicted_value_capped:.2f}%", (future_date, predicted_value_capped),
                    textcoords="offset points", xytext=(0,10), ha='center', va='bottom', color='orange')

        ax.plot(future_dates, predicted_values, marker='o', linestyle='--', color='orange', label='Projection')
        ax.legend()
        st.pyplot(fig)

        recent_data = df_filtered[df_filtered['Date'] >= df_filtered['Date'].max() - pd.DateOffset(months=2)]

        if not recent_data.empty:

            feature_cols = [col for col in df.columns if col not in targets + exclude_columns]
            features_recent = recent_data[feature_cols].apply(pd.to_numeric, errors='coerce')
            features_recent = features_recent.select_dtypes(include=[np.number])
            target_recent = recent_data[target]

            features_recent = features_recent.fillna(0)
            target_recent = target_recent.fillna(0)

            correlations = features_recent.corrwith(target_recent)
            correlations_abs = correlations.abs()
            top_7_features = correlations_abs.sort_values(ascending=False).head(7)

            with st.expander(f"Top 7 KPIs impacting {indicator_mapping[target]}", expanded=False):
                st.write(f"Top 7 KPIs contributing to the {trend_direction.lower()} trend in {indicator_mapping[target]}:")
                st.write(f"Predictive Model: {model_description}")

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

                    kpi_correlation = correlations[feature]
                    st.write(f"Impact on {indicator_mapping[target]}: Correlation = {kpi_correlation:.2f}")

                    pp_impact = kpi_correlation * change_pp
                    st.write(f"Estimated impact on {indicator_mapping[target]} over next 3 months: {pp_impact:.2f} pp")
                    st.write("---")

    if projections:
        projection_df = pd.DataFrame(projections).T
        projection_df = projection_df[['Current 3-Month Average (%)', 'Predicted Value in 3 Months (%)', 'Change (pp)', 'Trend']]

        numeric_columns = ['Current 3-Month Average (%)', 'Predicted Value in 3 Months (%)', 'Change (pp)']
        projection_df_style = projection_df.style.format({col: "{:.2f}" for col in numeric_columns})

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

def display_all_stores_customer_journey(df, column_metadata, date_range):
    st.title("All Stores: Customer Journey")

    # Filter data based on date range
    start_date, end_date = date_range
    try:
        df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['month'].astype(str), format='%Y-%m')
    except Exception as e:
        st.error(f"Error parsing 'month' and 'Year' columns into datetime: {e}")
        return

    df_filtered = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))].copy()

    if df_filtered.empty:
        st.warning("No data available for the selected date range.")
        return

    # Add this code to display the date information**
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    st.write(f"Data from **{start_date_str}** to **{end_date_str}** (use the date selector in the sidebar to change the date range).")

    # Prepare feature columns
    exclude_columns = ['establishment_name', 'month', 'Year', 'Date', 'satisfaction', 'value_for_money', 'return_probability', 'nps']
    feature_columns = [col for col in df_filtered.columns if col not in exclude_columns]
    features_filtered = df_filtered[feature_columns]

    # Convert features to numeric
    features_converted = features_filtered.apply(pd.to_numeric, errors='coerce')
    features_converted = features_converted.select_dtypes(include=[np.number])

    # **Remove columns with all missing values**
    features_converted = features_converted.dropna(axis=1, how='all')

    if features_converted.empty:
        st.error("No numeric features available for the selected date range.")
        return

    # Handle missing values in features_converted
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='most_frequent')
    X_filtered_imputed = pd.DataFrame(imputer.fit_transform(features_converted), columns=features_converted.columns)

    # Generate combined data for Customer Journey
    df_combined = plot_customer_journey(X_filtered_imputed, column_metadata)

    if df_combined is None:
        st.error("Failed to generate Customer Journey data.")
        return

    st.write("### Text Representation of Customer Journey")
    text_representation(df_combined)


def display_individual_store_customer_journey(df, column_metadata, date_range):
    st.title("Individual Store: Customer Journey")

    # Store Selector
    selected_stores = select_stores(df, selector_key="individual_customer_journey_store_selector")
    if not selected_stores:
        st.warning("Please select at least one store to view the Customer Journey.")
        return

    # Convert start_date and end_date to pd.Timestamp
    start_date, end_date = date_range
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Ensure 'Date' column exists
    try:
        df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['month'].astype(str), format='%Y-%m')
    except Exception as e:
        st.error(f"Error parsing 'month' and 'Year' columns into datetime: {e}")
        return

    # Filter data based on selected stores and date range
    df_filtered = df[df['establishment_name'].isin(selected_stores) & 
                     (df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()

    if df_filtered.empty:
        st.warning("No data available for the selected store(s) and date range.")
        return

    # Add this code to display the date information**
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    st.write(f"Data from **{start_date_str}** to **{end_date_str}** (use the date selector in the sidebar to change the date range).")


    # Prepare feature columns using the predefined exclude_columns
    # Also, exclude the target variables if they are not already in exclude_columns
    target_columns = ['satisfaction', 'value_for_money', 'return_probability', 'nps']
    all_exclude_columns = exclude_columns + target_columns
    feature_columns = [col for col in df_filtered.columns if col not in all_exclude_columns]
    features_filtered = df_filtered[feature_columns]

    # Convert features to numeric
    features_converted = features_filtered.apply(pd.to_numeric, errors='coerce')
    features_converted = features_converted.select_dtypes(include=[np.number])

    # Remove columns with all missing values
    features_converted = features_converted.dropna(axis=1, how='all')

    if features_converted.empty:
        st.error("No numeric features available for the selected store(s) and date range.")
        return

    # Handle missing values in features_converted
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='most_frequent')
    X_filtered_imputed = pd.DataFrame(imputer.fit_transform(features_converted), 
                                      columns=features_converted.columns)

    # Generate combined data for Customer Journey
    df_combined = plot_customer_journey(X_filtered_imputed, column_metadata)

    if df_combined is None:
        st.error("Failed to generate Customer Journey data.")
        return

    st.write("### Customer Journey Flow for Selected Store(s)")
    # No need to plot again as plot_customer_journey already handles plotting

    st.write("### Text Representation of Customer Journey")
    text_representation(df_combined)


def plot_customer_journey(df_data, column_metadata):
    # Calculate percentage of 1s for each KPI, ignoring blank values
    kpi_percentages = df_data.apply(lambda col: col.dropna().mean() * 100)
    kpi_percentages = kpi_percentages.reset_index()
    kpi_percentages.columns = ['Column Name', 'Percentage']

    # Combine metadata with KPI percentages
    df_combined = column_metadata.merge(kpi_percentages, on='Column Name', how='left')

    # Debugging: Check if 'Moment' exists
    st.write("### Combined Data Columns:")
    st.write(df_combined.columns)

    # Verify 'Moment' column exists
    if 'Moment' not in df_combined.columns:
        st.error("The 'Moment' column is missing after merging. Please check your column_metadata.")
        return None

    # Order moments by Step Number
    df_combined['Step Number'] = pd.to_numeric(df_combined['Step Number'], errors='coerce')
    df_combined = df_combined.sort_values('Step Number')

    # Plotting
    fig_height = max(20, 4 * len(df_combined['Moment'].unique()))
    fig_width = 20
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.05)

    for line_num, moment in enumerate(reversed(df_combined['Moment'].unique())):
        moment_data = df_combined[df_combined['Moment'] == moment]
        scores = moment_data['Percentage'].values
        labels = moment_data['Column Name'].values
        positions = range(len(scores))

        y_position = line_num * 2

        ax.plot(positions, [y_position] * len(scores), linestyle='-', color='gray', linewidth=3, alpha=0.7)

        for i, (pos, score, label) in enumerate(zip(positions, scores, labels)):
            color = plt.cm.viridis(score / 100)  # Color gradient based on score
            ax.scatter(pos, y_position, color=color, s=300, zorder=5, marker='o')
            ax.text(pos, y_position + 0.4, f'{score:.0f}%', fontsize=20, ha='center', va='bottom')
            ax.text(pos, y_position - 0.4, textwrap.fill(label, width=20), fontsize=13, ha='center', va='top')

        avg_score = moment_data['Percentage'].mean()
        ax.text(-0.5, y_position, f"{moment} ({avg_score:.0f}%)", fontsize=16, ha='right', va='center', fontweight='bold')

    ax.set_title("Customer Journey Flow", fontsize=18, fontweight='bold', pad=20)
    ax.set_yticks([])
    ax.set_ylim(-1, len(df_combined['Moment'].unique()) * 2)

    legend_labels = ['High Friction (< 60%)', 'Medium Friction (60%-75%)', 'Goal Achieved (> 75%)']
    legend_colors = [plt.cm.viridis(0.25), plt.cm.viridis(0.6), plt.cm.viridis(0.9)]
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=10) for c in legend_colors]
    ax.legend(handles, legend_labels, title="Score Breakdown", loc='upper left', bbox_to_anchor=(1, 1))

    st.pyplot(fig)

    return df_combined

def text_representation(df_combined):
    if df_combined is None:
        st.error("No Customer Journey data available.")
        return

    st.header("Customer Journey Flow - Text Representation")
    
    unique_moments = df_combined['Moment'].unique()
    
    for moment in unique_moments:
        moment_data = df_combined[df_combined['Moment'] == moment]
        avg_score = moment_data['Percentage'].mean()
        
        st.subheader(f"{moment} (Average: {avg_score:.0f}%)")
        
        for _, row in moment_data.iterrows():
            kpi = row['Column Name']
            score = row['Percentage']
            if score < 60:
                dot_color = "#FFCCCB"  # Light red
            elif 60 <= score < 75:
                dot_color = "#FFFFA1"  # Light yellow
            else:
                dot_color = "#90EE90"  # Light green
            
            st.markdown(f'<span style="color:{dot_color};">●</span> {kpi}: {score:.0f}%', unsafe_allow_html=True)
        
        st.write("---")

def get_worst_moment(df_combined):
    moment_averages = df_combined.groupby('Moment')['Percentage'].mean()
    worst_moment = moment_averages.idxmin()
    worst_score = moment_averages.min()
    return worst_moment, worst_score

def get_okr_summary(df_combined):
    okrs = ['satisfaction', 'value_for_money', 'return_probability', 'nps']
    okr_summary = df_combined[df_combined['Column Name'].isin(okrs)].set_index('Column Name')['Percentage']
    return okr_summary


# Test OpenAI Connection
def test_openai_connection():
    """
    Test the OpenAI connection by making a simple API call.
    """
    try:
        response = openai.Engine.list()  # Lists available engines
        st.success("Successfully connected to OpenAI!")
        st.write("Available Engines:")
        for engine in response['data']:
            st.write(f"- {engine['id']}")
    except Exception as e:
        st.error(f"Failed to connect to OpenAI: {e}")
        st.stop()

## On/Off Target Analysis (What Ifs)
def display_what_if_analysis(df, X_imputed, y_imputed, targets, indicator_mapping):
    st.title("All Stores: What If (Impact Analysis)")

     # Add this code to display the warning message**
    st.markdown(
        """
        <div style="background-color:#f8d7da; padding: 10px; border-radius: 5px; font-size: 12px;">
            <strong>Note:</strong> If you encounter a 'StopIteration' error or see 'NaN%' in the results, it may indicate that the sample size is too small for reliable calculations. If that's the case, please interpret the data with caution.
        </div>
        """,
        unsafe_allow_html=True
    )

    # Calculate impact for each feature
    impact_data = []
    for feature in X_imputed.columns:
        feature_impact = {}
        for target in targets:
            on_target = y_imputed[target][X_imputed[feature] == 1].mean() * 100
            off_target = y_imputed[target][X_imputed[feature] == 0].mean() * 100
            gap = on_target - off_target
            correlation = X_imputed[feature].corr(y_imputed[target])
            feature_impact[target] = {
                'on_target': on_target,
                'off_target': off_target,
                'gap': gap,
                'correlation': correlation
            }
        impact_data.append((feature, feature_impact))

    # Add sorting options
    st.subheader("Sort Options")
    sort_col1, sort_col2 = st.columns(2)
    with sort_col1:
        sort_method = st.selectbox("Sort by:", ["Gap (Default)", "Correlation"])
    with sort_col2:
        if sort_method == "Correlation":
            sort_target = st.selectbox("Target for correlation:", targets, format_func=lambda x: indicator_mapping[x])
        else:
            sort_target = "satisfaction"  # Default for gap sorting

    # Sort the impact_data based on user selection
    if sort_method == "Gap (Default)":
        impact_data.sort(key=lambda x: abs(x[1][sort_target]['gap']), reverse=True)
    else:  # Correlation
        impact_data.sort(key=lambda x: abs(x[1][sort_target]['correlation']), reverse=True)

    # Display impact analysis for each feature
    for feature, impact in impact_data:
        st.write("---")
        st.subheader(f":chart_with_upwards_trend: {feature}")

        for target in targets:
            # **Construct and display the explanatory sentence**
            on_target_mean_formatted = f"{impact[target]['on_target']:.2f}%"
            off_target_mean_formatted = f"{impact[target]['off_target']:.2f}%"
            gap_formatted = f"{impact[target]['gap']:.2f}%"

            explanatory_sentence = (
                f"Customers who are **highly satisfied** (On-Target) with **{feature}** "
                f"have an average {indicator_mapping[target]} score of {on_target_mean_formatted}, "
                f"while less satisfied customers with the same metric have a score of only {off_target_mean_formatted}."
            )

            st.write(explanatory_sentence)

            with st.expander(f"{indicator_mapping[target]} Analysis", expanded=True):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("On Target", on_target_mean_formatted)
                
                with col2:
                    st.metric("Off Target", off_target_mean_formatted)
                
                with col3:
                    gap = impact[target]['gap']
                    delta_color = "normal" if gap > 0 else "inverse"
                    st.metric("Gap On-Off Target", gap_formatted, delta=gap_formatted, delta_color=delta_color)
                
                correlation = impact[target]['correlation']
                corr_color = "green" if correlation > 0 else "red"
                st.markdown(
                    f"**Correlation:** <span style='color:{corr_color}'>{correlation:.2f}</span>",
                    unsafe_allow_html=True
                )

        # Provide a summary of the feature's impact
        st.write("### Summary")
        max_gap = max(impact.values(), key=lambda x: abs(x['gap']))
        max_gap_target = next(t for t, v in impact.items() if v['gap'] == max_gap['gap'])
        st.write(
            f"**{feature}** has the largest impact on **{indicator_mapping[max_gap_target]}** "
            f"with a gap of {max_gap['gap']:.2f}% between on-target and off-target performance."
        )



def display_individual_what_if_analysis(df, X_imputed, y_imputed, targets, indicator_mapping):
    st.title("Individual Store: What If (Impact Analysis)")

    # Store selection
    selected_stores = select_stores(df, selector_key="individual_what_if_store_selector")
    
    if not selected_stores:
        st.warning("Please select at least one store to proceed with the analysis.")
        return

    # Add this code to display the warning message**
    st.markdown(
        """
        <div style="background-color:#f8d7da; padding: 10px; border-radius: 5px; font-size: 12px;">
            <strong>Note:</strong> If you encounter a 'StopIteration' error or see 'NaN%' in the results, it may indicate that the sample size is too small for reliable calculations. If that's the case, please interpret the data with caution.
        </div>
        """,
        unsafe_allow_html=True
    )

    # Filter data based on selected stores
    store_mask = df['establishment_name'].isin(selected_stores)
    X_imputed_filtered = X_imputed[store_mask]
    y_imputed_filtered = y_imputed[store_mask]

    st.write(f"**Analysis for selected store(s):** {', '.join(selected_stores)}")

    # Calculate impact for each feature
    impact_data = []
    for feature in X_imputed_filtered.columns:
        feature_impact = {}
        for target in targets:
            on_target = y_imputed_filtered[target][X_imputed_filtered[feature] == 1].mean() * 100
            off_target = y_imputed_filtered[target][X_imputed_filtered[feature] == 0].mean() * 100
            gap = on_target - off_target
            correlation = X_imputed_filtered[feature].corr(y_imputed_filtered[target])
            feature_impact[target] = {
                'on_target': on_target,
                'off_target': off_target,
                'gap': gap,
                'correlation': correlation
            }
        impact_data.append((feature, feature_impact))

    # Add sorting options
    st.subheader("Sort Options")
    sort_col1, sort_col2 = st.columns(2)
    with sort_col1:
        sort_method = st.selectbox("Sort by:", ["Gap (Default)", "Correlation"], key="individual_sort_method")
    with sort_col2:
        if sort_method == "Correlation":
            sort_target = st.selectbox(
                "Target for correlation:",
                targets,
                format_func=lambda x: indicator_mapping[x],
                key="individual_sort_target"
            )
        else:
            sort_target = "satisfaction"  # Default for gap sorting

    # Sort the impact_data based on user selection
    if sort_method == "Gap (Default)":
        impact_data.sort(key=lambda x: abs(x[1][sort_target]['gap']), reverse=True)
    else:  # Correlation
        impact_data.sort(key=lambda x: abs(x[1][sort_target]['correlation']), reverse=True)

    # Display impact analysis for each feature
    for feature, impact in impact_data:
        st.write("---")
        st.subheader(f":chart_with_upwards_trend: {feature}")
        
        for target in targets:
            # **Construct and display the explanatory sentence**
            on_target_mean_formatted = f"{impact[target]['on_target']:.2f}%"
            off_target_mean_formatted = f"{impact[target]['off_target']:.2f}%"
            gap_formatted = f"{impact[target]['gap']:.2f}%"
            
            explanatory_sentence = (
                f"Customers who are **highly satisfied** (On-Target) with **{feature}** "
                f"have an average {indicator_mapping[target]} score of {on_target_mean_formatted}, "
                f"while less satisfied customers with the same metric have a score of only {off_target_mean_formatted}."
            )
            
            st.write(explanatory_sentence)
            
            with st.expander(f"{indicator_mapping[target]} Analysis", expanded=True):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("On Target", on_target_mean_formatted)
                
                with col2:
                    st.metric("Off Target", off_target_mean_formatted)
                
                with col3:
                    gap = impact[target]['gap']
                    delta_color = "normal" if gap > 0 else "inverse"
                    st.metric(
                        "Gap On-Off Target",
                        gap_formatted,
                        delta=gap_formatted,
                        delta_color=delta_color
                    )
                
                correlation = impact[target]['correlation']
                corr_color = "green" if correlation > 0 else "red"
                st.markdown(
                    f"**Correlation:** <span style='color:{corr_color}'>{correlation:.2f}</span>",
                    unsafe_allow_html=True
                )

        # Provide a summary of the feature's impact
        st.write("### Summary")
        max_gap = max(impact.values(), key=lambda x: abs(x['gap']))
        max_gap_target = next(t for t, v in impact.items() if v['gap'] == max_gap['gap'])
        st.write(
            f"**{feature}** has the largest impact on **{indicator_mapping[max_gap_target]}** "
            f"with a gap of {max_gap['gap']:.2f}% between on-target and off-target performance."
        )


def create_pivot_table(df, x_axis, y_axis):
    if x_axis == 'establishment_name':
        pivot_table = pd.pivot_table(df, values=y_axis, index=x_axis, 
                                     aggfunc=[lambda x: (x == 1).mean() * 100, lambda x: (x == 0).mean() * 100])
        pivot_table.columns = ['On-Target (%)', 'Off-Target (%)']
        pivot_table.index.name = f"**{x_axis}**"
    elif y_axis == 'establishment_name':
        pivot_table = pd.pivot_table(df, values=x_axis, index=y_axis, 
                                     aggfunc=[lambda x: (x == 1).mean() * 100, lambda x: (x == 0).mean() * 100])
        pivot_table.columns = ['On-Target (%)', 'Off-Target (%)']
        pivot_table.index.name = f"**{y_axis}**"
    else:
        # For non-one-dimensional X-axis
        df_temp = df.copy()
        df_temp[x_axis] = df_temp[x_axis].map({0: "Off Target", 1: "On Target"})
        pivot_table = pd.pivot_table(df_temp, values=y_axis, index=x_axis, columns=None,
                                     aggfunc=lambda x: (x == 1).mean() * 100)
        pivot_table = pivot_table.reindex(["On Target", "Off Target"])
        pivot_table.columns = [f'**{y_axis}** (%)']
        pivot_table.index.name = f"**{x_axis}**"

    return pivot_table.round(2)

def display_all_stores_pivot_table(df):
    st.title("All Stores: Pivot Table")

    # Combine Month and Year into Date
    df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['month'].astype(str), format='%Y-%m')

    # Date range filter in sidebar
    st.sidebar.header("Date Range Selection")
    min_date = df['Date'].min()
    max_date = df['Date'].max()
    start_date = st.sidebar.date_input("Start date", min_date, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("End date", max_date, min_value=min_date, max_value=max_date)

    if start_date > end_date:
        st.error("Error: End date must be after start date.")
        return

    # Filter dataframe based on selected date range
    df_filtered = df[(df['Date'] >= pd.Timestamp(start_date)) & (df['Date'] <= pd.Timestamp(end_date))]

    # Add this code to display the date information**
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    st.write(f"Data from **{start_date_str}** to **{end_date_str}** (use the date selector in the sidebar to change the date range).")
    
    
    # Get all column names excluding 'Date', 'month', and 'Year'

    all_columns = ['establishment_name'] + [col for col in df.columns if col not in exclude_columns]


    # Select X and Y axes
    col1, col2 = st.columns(2)
    with col1:
        x_axis = st.selectbox("Select X-axis:", all_columns, key="all_stores_pivot_x_axis")
    with col2:
        y_axis = st.selectbox("Select Y-axis:", [col for col in all_columns if col != x_axis], key="all_stores_pivot_y_axis")

    # Create pivot table
    pivot_table = create_pivot_table(df_filtered, x_axis, y_axis)

    # Display the pivot table with color coding
    st.dataframe(pivot_table.style
                 .format("{:.2f}%")
                 .applymap(lambda x: 'color: green', subset=pivot_table.columns[pivot_table.columns.str.contains('On-Target')])
                 .applymap(lambda x: 'color: red', subset=pivot_table.columns[pivot_table.columns.str.contains('Off-Target')]))

    # Add download button for CSV
    csv = pivot_table.to_csv().encode('utf-8')
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name=f'all_stores_pivot_table_{x_axis}_vs_{y_axis}.csv',
        mime='text/csv',
    )

def display_individual_pivot_table(df):
    st.title("Individual Store: Pivot Table")

    # Combine Month and Year into Date
    df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['month'].astype(str), format='%Y-%m')

    # Store selection
    selected_stores = select_stores(df, selector_key="individual_pivot_store_selector")
    
    if not selected_stores:
        st.warning("Please select at least one store to proceed with the analysis.")
        return

    # Filter data based on selected stores
    df_filtered = df[df['establishment_name'].isin(selected_stores)]

    st.write(f"**Analysis for selected store(s):** {', '.join(selected_stores)}")

    # Date range filter in sidebar
    st.sidebar.header("Date Range Selection")
    min_date = df_filtered['Date'].min()
    max_date = df_filtered['Date'].max()
    start_date = st.sidebar.date_input("Start date", min_date, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("End date", max_date, min_value=min_date, max_value=max_date)

    if start_date > end_date:
        st.error("Error: End date must be after start date.")
        return

    # Add this code to display the date information**
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    st.write(f"Data from **{start_date_str}** to **{end_date_str}** (use the date selector in the sidebar to change the date range).")

    # Further filter dataframe based on selected date range
    df_filtered = df_filtered[(df_filtered['Date'] >= pd.Timestamp(start_date)) & (df_filtered['Date'] <= pd.Timestamp(end_date))]

    # Get all column names excluding 'Date', 'month', and 'Year'
    all_columns = ['establishment_name'] + [col for col in df.columns if col not in ['establishment_name', 'month', 'Year', 'Date']]

    # Select X and Y axes
    col1, col2 = st.columns(2)
    with col1:
        x_axis = st.selectbox("Select X-axis:", all_columns, key="individual_pivot_x_axis")
    with col2:
        y_axis = st.selectbox("Select Y-axis:", [col for col in all_columns if col != x_axis], key="individual_pivot_y_axis")

    # Create pivot table
    pivot_table = create_pivot_table(df_filtered, x_axis, y_axis)

    # Display the pivot table with color coding
    st.dataframe(pivot_table.style
                 .format("{:.2f}%")
                 .applymap(lambda x: 'color: green', subset=pivot_table.columns[pivot_table.columns.str.contains('On-Target')])
                 .applymap(lambda x: 'color: red', subset=pivot_table.columns[pivot_table.columns.str.contains('Off-Target')]))

    # Add download button for CSV
    csv = pivot_table.to_csv().encode('utf-8')
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name=f'individual_pivot_table_{x_axis}_vs_{y_axis}.csv',
        mime='text/csv',
    )


def display_ask_me_anything(df):
    st.title("Ask Me Anything")

    # Combine all comment columns
    comment_columns = ['Comments 1', 'Comments 2', 'Comments 3', 'Comments 4', 'Comments 5']
    all_comments = df[comment_columns].fillna('').agg(' '.join, axis=1)

    # One-word summary
    one_word_summary_prompt = f"Based on the following customer comments, summarize this brand or location in one word: {all_comments.str.cat(sep=' ')}"
    one_word_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an experienced restauranteur providing a one-word summary."},
            {"role": "user", "content": one_word_summary_prompt}
        ],
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.5,
    )
    one_word_summary = one_word_response.choices[0].message['content'].strip()

    st.subheader("This brand or location in one word:")
    st.write(one_word_summary)

    # User question input
    user_question = st.text_input("Ask me anything CX about this brand:")

    if user_question:
        prompt = f"Acting as an experienced restauranteur, and considering exclusively the following comments, please answer this question, but base your answer in most relevant topics, showing the % of times this topic or group of topic is mentioned in the comments: {user_question}\n\nComments: {all_comments.str.cat(sep=' ')}"
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an experienced restauranteur providing insights based on customer comments."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            n=1,
            stop=None,
            temperature=0.7,
        )

        answer = response.choices[0].message['content'].strip()
        st.write("Answer:", answer)


def display_reputation_index(df):
    """
    Display the Reputation Index analysis for all stores based on the 'nps2' column.
    """
    st.header("All Stores: Reputation Index")
    
    # Check if 'nps2' column exists
    if 'nps2' not in df.columns:
        st.error("The 'nps2' column is missing from the dataset.")
        return

    # Calculate counts
    total_responses = len(df)
    promoters = df['nps2'].value_counts().get('Promoter', 0)
    detractors = df['nps2'].value_counts().get('Detractor', 0)
    neutrals = df['nps2'].value_counts().get('Neutral', 0)
    
    # Calculate percentages
    promoters_pct = (promoters / total_responses) * 100
    detractors_pct = (detractors / total_responses) * 100
    neutrals_pct = (neutrals / total_responses) * 100
    
    # Display average percentages using metrics
    st.subheader("Average Ratings")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Promoters", f"{promoters_pct:.2f}%")
    with col2:
        st.metric("Detractors", f"{detractors_pct:.2f}%")
    with col3:
        st.metric("Neutrals", f"{neutrals_pct:.2f}%")
    
    # Calculate Net Promoter Score (NPS)
    nps = promoters_pct - detractors_pct
    st.subheader("Net Promoter Score (NPS)")
    st.metric("NPS", f"{nps:.2f}")
    
    st.markdown("---")
    
    # Word of Mouth Section
    st.subheader("Word of Mouth")
    
    # Perform the calculations
    word_of_mouth_promoters = promoters_pct * 1.15
    word_of_mouth_detractors = detractors_pct * 4.60
    
    # Reputation calculation
    reputation = word_of_mouth_promoters - word_of_mouth_detractors

    # Display the Word of Mouth metrics
    col4, col5 = st.columns(2)
    
    with col4:
        st.metric("Promoters * 1.15", f"{word_of_mouth_promoters:.2f}")
    with col5:
        st.metric("Detractors * 4.60", f"{word_of_mouth_detractors:.2f}")

    # Display Reputation
    st.metric("Reputation (Word of Mouth)", f"{reputation:.2f}", delta_color="inverse")
    



    # === Added Expander for OpenAI Analysis ===
    with st.expander("**How to Fix or Improve It?** AI Generated based on you customer's perception"):
        # Filter only 'Detractors'
        detractors_df = df[df['nps2'] == 'Detractor']
        
        # Check if there are any detractors
        if detractors_df.empty:
            st.write("No Detractor comments available.")
        else:
            # Get the comments columns
            comments_columns = ['Comments 1', 'Comments 2', 'Comments 3', 'Comments 4', 'Comments 5']
            # Ensure the comments columns exist in the dataframe
            comments_columns = [col for col in comments_columns if col in detractors_df.columns]
            
            # Check if there are any comments columns
            if not comments_columns:
                st.write("No comments available for Detractors.")
            else:
                # Concatenate the comments into a single string
                comments_series = detractors_df[comments_columns].astype(str).agg(' '.join, axis=1)
                # Combine all comments into one text
                all_comments = ' '.join(comments_series.tolist())
                
                # Prepare the prompt
                prompt = (
                    "Based on the comments provided below, and assuming the role of an experienced business analyst, "
                    "identify the most frequent complaints among customers that lead to them becoming Detractors. "
                    "Please categorize the complaints by theme, provide the percentage of times each complaint is mentioned, "
                    "and highlight any patterns or insights that may explain why these issues are driving negative feedback.\n\n"
                    f"Comments:\n{all_comments}"
                )
                
                # Display a progress spinner
                with st.spinner("Analyzing comments with OpenAI..."):
                    try:
                        # Call the OpenAI API
                        response = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "user", "content": prompt}
                            ],
                            max_tokens=500,
                            temperature=0.7,
                        )
                        
                        # Extract the response
                        answer = response['choices'][0]['message']['content']
                        
                        # Display the response
                        st.write(answer)
                    except Exception as e:
                        st.error(f"An error occurred while communicating with OpenAI: {e}")
    
    st.markdown("---")


    

    # === Time-Series Analysis Section ===

    st.markdown("---")
    st.subheader("Time-Series Analysis of Reputation")

    # Ensure 'Date' column is in datetime format
    if 'Date' not in df.columns:
        try:
            df['Date'] = pd.to_datetime(
                df['Year'].astype(str) + '-' + df['month'].astype(str), format='%Y-%m'
            )
        except Exception as e:
            st.error(f"Error parsing 'month' and 'Year' columns into datetime: {e}")
            return
    else:
        df['Date'] = pd.to_datetime(df['Date'])

    # Aggregate data monthly using calculate_monthly_metrics
    monthly_data = df.groupby(df['Date'].dt.to_period('M')).apply(calculate_monthly_metrics).reset_index()
    monthly_data['Date'] = monthly_data['Date'].dt.to_timestamp()

    # Sort the data by Date
    monthly_data = monthly_data.sort_values('Date')

    # Calculate moving averages for Reputation
    monthly_data['Reputation_6M_MA'] = monthly_data['Reputation'].rolling(window=6).mean()
    monthly_data['Reputation_3M_MA'] = monthly_data['Reputation'].rolling(window=3).mean()

    # Forecast the next 3 months for Reputation
    forecast_df = forecast_reputation(monthly_data)

    # Combine historical and forecasted data
    combined_data = pd.concat([monthly_data, forecast_df], ignore_index=True, sort=False)

    # Plot the data
    plot_reputation_time_series(combined_data)

    # Display the table below the chart
    display_reputation_table(monthly_data)

# Helper function to calculate monthly metrics including Reputation
def calculate_monthly_metrics(group):
    total_responses = len(group)
    promoters = group['nps2'].value_counts().get('Promoter', 0)
    detractors = group['nps2'].value_counts().get('Detractor', 0)
    promoters_pct = (promoters / total_responses) * 100
    detractors_pct = (detractors / total_responses) * 100
    nps = promoters_pct - detractors_pct

    # Word of Mouth calculations
    word_of_mouth_promoters = promoters_pct * 1.15
    word_of_mouth_detractors = detractors_pct * 4.60

    # Reputation calculation
    reputation = word_of_mouth_promoters - word_of_mouth_detractors

    return pd.Series({
        'Promoters_pct': promoters_pct,
        'Detractors_pct': detractors_pct,
        'NPS': nps,
        'Reputation': reputation
    })

# Helper function to forecast Reputation for the next 3 months
def forecast_reputation(monthly_data):
    # Create future dates starting from the next month after the last date in historical data
    last_date = monthly_data['Date'].max()
    forecast_index = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=3, freq='MS')

    # Calculate averages for the last 6 and 3 months
    forecast_6m_avg = monthly_data['Reputation'].tail(6).mean()
    forecast_3m_avg = monthly_data['Reputation'].tail(3).mean()

    # Create forecast DataFrame with both averages
    forecast_df = pd.DataFrame({
        'Date': forecast_index,
        'Reputation_6M_Forecast': [forecast_6m_avg] * 3,
        'Reputation_3M_Forecast': [forecast_3m_avg] * 3,
        'Forecast_Type': 'Average'
    })

    # No moving averages for forecasted data
    forecast_df['Reputation_6M_MA'] = None
    forecast_df['Reputation_3M_MA'] = None

    return forecast_df

# Helper function to plot Reputation time series
def plot_reputation_time_series(data, store_name=None):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    plt.style.use('default')  # Set the default style (white background)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot historical Reputation
    historical_data = data.dropna(subset=['Reputation'])
    ax.plot(historical_data['Date'], historical_data['Reputation'], label='Reputation', marker='o', linestyle='-')

    # Plot moving averages
    ax.plot(historical_data['Date'], historical_data['Reputation_6M_MA'], label='6-Month MA', linestyle='--')
    ax.plot(historical_data['Date'], historical_data['Reputation_3M_MA'], label='3-Month MA', linestyle='-.')

    # Plot forecasted Reputation based on averages
    forecast_data = data[data['Forecast_Type'] == 'Average']
    if not forecast_data.empty:
        ax.plot(forecast_data['Date'], forecast_data['Reputation_6M_Forecast'], label='6M Average Forecast', marker='x', linestyle='--', color='blue')
        ax.plot(forecast_data['Date'], forecast_data['Reputation_3M_Forecast'], label='3M Average Forecast', marker='x', linestyle='--', color='orange')

        # Annotate forecasted data points with "Positive" or "Negative" and expected Reputation value
        for idx, row in forecast_data.iterrows():
            date = row['Date']

            # For 6M forecast
            reputation_6m = row['Reputation_6M_Forecast']
            sentiment_6m = 'Positive' if reputation_6m >= 0 else 'Negative'
            ax.annotate(f"{sentiment_6m}\n{reputation_6m:.2f}",
                        xy=(date, reputation_6m),
                        xytext=(0, 15),
                        textcoords='offset points',
                        ha='center',
                        fontsize=8,
                        color='blue')

            # For 3M forecast
            reputation_3m = row['Reputation_3M_Forecast']
            sentiment_3m = 'Positive' if reputation_3m >= 0 else 'Negative'
            ax.annotate(f"{sentiment_3m}\n{reputation_3m:.2f}",
                        xy=(date, reputation_3m),
                        xytext=(0, -30),
                        textcoords='offset points',
                        ha='center',
                        fontsize=8,
                        color='orange')

    # Annotate each data point with "Positive" or "Negative"
    for idx, row in historical_data.iterrows():
        date = row['Date']
        reputation_value = row['Reputation']
        sentiment = 'Positive' if reputation_value >= 0 else 'Negative'
        vertical_offset = 5 if reputation_value >= 0 else -15
        ax.annotate(sentiment,
                    xy=(date, reputation_value),
                    xytext=(0, vertical_offset),
                    textcoords='offset points',
                    ha='center',
                    fontsize=8,
                    color='green' if reputation_value >= 0 else 'red')

    # Draw zero line
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)

    # Adjust the legend
    ax.legend()

    # Formatting x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=45)

    # Update Y-axis label
    if store_name:
        ax.set_title(f'Reputation Time Series with Forecast for {store_name}')
    else:
        ax.set_title('Reputation Time Series with Forecast')
    ax.set_xlabel('Date')
    ax.set_ylabel('Reputation: Word of Mouth')

    # Set the background to white
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    st.pyplot(fig)



# Helper function to display the Reputation table
def display_reputation_table(monthly_data):
    # Prepare the data for the table
    table_data = monthly_data[['Date', 'Reputation', 'Promoters_pct', 'Detractors_pct']].copy()
    table_data['Reputation'] = table_data['Reputation'].round(2)
    table_data['Promoters_pct'] = table_data['Promoters_pct'].round(2)
    table_data['Detractors_pct'] = table_data['Detractors_pct'].round(2)

    # Add Positive/Negative column based on Reputation
    table_data['Sentiment'] = table_data['Reputation'].apply(lambda x: 'Positive' if x >= 0 else 'Negative')

    # Rename columns for display
    table_data = table_data.rename(columns={
        'Date': 'Month',
        'Reputation': 'Word of Mouth',
        'Promoters_pct': '% of Promoters',
        'Detractors_pct': '% of Detractors'
    })

    # Compute the 3-Months Average Reputation
    avg_3m_reputation = table_data['Word of Mouth'].tail(3).mean()
    sentiment_3m = 'Positive' if avg_3m_reputation >= 0 else 'Negative'

    # Compute the 6-Months Average Reputation
    avg_6m_reputation = table_data['Word of Mouth'].tail(6).mean()
    sentiment_6m = 'Positive' if avg_6m_reputation >= 0 else 'Negative'

    # Create DataFrames for the average rows
    avg_3m_row = pd.DataFrame({
        'Month': ['3-Months Average'],
        'Word of Mouth': [round(avg_3m_reputation, 2)],
        '% of Promoters': [None],
        '% of Detractors': [None],
        'Sentiment': [sentiment_3m]
    })

    avg_6m_row = pd.DataFrame({
        'Month': ['6-Months Average'],
        'Word of Mouth': [round(avg_6m_reputation, 2)],
        '% of Promoters': [None],
        '% of Detractors': [None],
        'Sentiment': [sentiment_6m]
    })

    # Append the average rows to the table data
    table_data = pd.concat([table_data, avg_3m_row, avg_6m_row], ignore_index=True)

    # Reset index to avoid issues with the styler
    table_data = table_data.reset_index(drop=True)

    # Define a function to color positive and negative values
    def color_positive_negative(val):
        if isinstance(val, (int, float)):
            color = 'green' if val >= 0 else 'red'
            return f'color: {color}'
        elif val == 'Positive':
            return 'color: green'
        elif val == 'Negative':
            return 'color: red'
        else:
            return ''

    # Apply the style to the dataframe
    def style_positive_negative(series):
        return ['color: green' if v >= 0 else 'color: red' if v < 0 else '' for v in series]

    def style_sentiment(series):
        return ['color: green' if v == 'Positive' else 'color: red' if v == 'Negative' else '' for v in series]

    styled_table = table_data.style.apply(style_positive_negative, subset=['Word of Mouth'])
    styled_table = styled_table.apply(style_sentiment, subset=['Sentiment'])

    # Display the table
    st.subheader("Monthly Reputation Metrics")
    st.write(styled_table.to_html(), unsafe_allow_html=True)

    st.write('**Word of Mouth** refers to the number of customers who will hear either positive or negative feedback about your brand or location for every 100 visits you receive.')

    st.write('This means that for every 100 customers who visit your store, a certain number will share their experience—good or bad—with others, influencing your brand\'s reputation.')

    st.write('It’s important to note that negative experiences tend to reverberate more strongly among other customers than positive ones. In other words, bad news travels fast. On average, **for every one negative experience (rated 1, 2, or 3 in satisfaction), it takes about twelve positive experiences to offset the damage caused by that single negative encounter.**')

    st.write('**Word of Mouth** translates these perceptions into numbers and helps forecast how your brand\'s reputation is performing over time, giving you insight into the potential long-term effects of both positive and negative customer experiences.')


def display_individual_store_reputation_index(df):
    """
    Display the Reputation Index analysis for individual stores based on the 'nps2' column.
    """
    st.header("Individual Store: Reputation Index")

    # Store Selector
    selected_stores = select_stores(df, selector_key="individual_reputation_index_store_selector")
    if not selected_stores:
        st.warning("Please select at least one store to view the Customer Journey.")
        return


    # Filter the dataframe based on selected stores
    if selected_stores:
        st.write(f"**Analysis for selected store(s):** {', '.join(selected_stores)}")
        store_df = df[df['establishment_name'].isin(selected_stores)].copy()
    else:
        st.write("Please select at least one store.")
        return


    
    # Check if 'nps2' column exists in the filtered data
    if 'nps2' not in store_df.columns:
        st.error("The 'nps2' column is missing from the dataset.")
        return

    # Check if the filtered dataframe is empty
    if store_df.empty:
        st.write("No data available for the selected store.")
        return
    
    # Calculate counts
    total_responses = len(store_df)
    promoters = store_df['nps2'].value_counts().get('Promoter', 0)
    detractors = store_df['nps2'].value_counts().get('Detractor', 0)
    neutrals = store_df['nps2'].value_counts().get('Neutral', 0)
    
    # Calculate percentages
    promoters_pct = (promoters / total_responses) * 100
    detractors_pct = (detractors / total_responses) * 100
    neutrals_pct = (neutrals / total_responses) * 100
    
    # Display average percentages using metrics
    st.subheader(f"Average Ratings for {selected_stores}")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Promoters", f"{promoters_pct:.2f}%")
    with col2:
        st.metric("Detractors", f"{detractors_pct:.2f}%")
    with col3:
        st.metric("Neutrals", f"{neutrals_pct:.2f}%")
    
    # Calculate Net Promoter Score (NPS)
    nps = promoters_pct - detractors_pct
    st.subheader("Net Promoter Score (NPS)")
    st.metric("NPS", f"{nps:.2f}")
    
    st.markdown("---")
    
    # Word of Mouth Section
    st.subheader("Word of Mouth")
    
    # Perform the calculations
    word_of_mouth_promoters = promoters_pct * 1.15
    word_of_mouth_detractors = detractors_pct * 4.60
    
    # Reputation calculation
    reputation = word_of_mouth_promoters - word_of_mouth_detractors

    # Display the Word of Mouth metrics
    col4, col5 = st.columns(2)
    
    with col4:
        st.metric("Promoters * 1.15", f"{word_of_mouth_promoters:.2f}")
    with col5:
        st.metric("Detractors * 4.60", f"{word_of_mouth_detractors:.2f}")
    
    # Display Reputation
    st.metric("Reputation (Word of Mouth)", f"{reputation:.2f}", delta_color="inverse")
    
    # === Added Expander for OpenAI Analysis ===
    with st.expander(f"**How to Fix or Improve {selected_stores}?** AI Generated based on your customer's perception"):
        # Filter only 'Detractors' in the store data
        detractors_df = store_df[store_df['nps2'] == 'Detractor']
        
        # Check if there are any detractors
        if detractors_df.empty:
            st.write("No Detractor comments available for this store.")
        else:
            # Get the comments columns
            comments_columns = ['Comments 1', 'Comments 2', 'Comments 3', 'Comments 4', 'Comments 5']
            # Ensure the comments columns exist in the dataframe
            comments_columns = [col for col in comments_columns if col in detractors_df.columns]
            
            # Check if there are any comments columns
            if not comments_columns:
                st.write("No comments available for Detractors.")
            else:
                # Concatenate the comments into a single string
                comments_series = detractors_df[comments_columns].astype(str).agg(' '.join, axis=1)
                # Combine all comments into one text
                all_comments = ' '.join(comments_series.tolist())
                
                # Prepare the prompt
                prompt = (
                    f"Based on the comments provided below for {selected_stores}, and assuming the role of an experienced business analyst, "
                    "identify the most frequent complaints among customers that lead to them becoming Detractors. "
                    "Please categorize the complaints by theme, provide the percentage of times each complaint is mentioned, "
                    "and highlight any patterns or insights that may explain why these issues are driving negative feedback.\n\n"
                    f"Comments:\n{all_comments}"
                )
                
                # OpenAI integration
                # The API key is already set in initialize_openai()
                
                # Display a progress spinner
                with st.spinner("Analyzing comments with OpenAI..."):
                    try:
                        # Call the OpenAI API
                        response = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "user", "content": prompt}
                            ],
                            max_tokens=500,
                            temperature=0.7,
                        )
                        
                        # Extract the response
                        answer = response['choices'][0]['message']['content']
                        
                        # Display the response
                        st.write(answer)
                    except Exception as e:
                        st.error(f"An error occurred while communicating with OpenAI: {e}")
    
    st.markdown("---")
    
    # === Time-Series Analysis Section ===

    st.markdown("---")
    st.subheader(f"Time-Series Analysis of Reputation for {selected_stores}")

    # Ensure 'Date' column is in datetime format
    if 'Date' not in store_df.columns:
        try:
            store_df['Date'] = pd.to_datetime(
                store_df['Year'].astype(str) + '-' + store_df['month'].astype(str), format='%Y-%m'
            )
        except Exception as e:
            st.error(f"Error parsing 'month' and 'Year' columns into datetime: {e}")
            return
    else:
        store_df['Date'] = pd.to_datetime(store_df['Date'])

    # Aggregate data monthly using calculate_monthly_metrics
    monthly_data = store_df.groupby(store_df['Date'].dt.to_period('M')).apply(calculate_monthly_metrics).reset_index()
    monthly_data['Date'] = monthly_data['Date'].dt.to_timestamp()

    # Check if monthly_data is empty
    if monthly_data.empty:
        st.write("No monthly data available for the selected store.")
        return

    # Sort the data by Date
    monthly_data = monthly_data.sort_values('Date')

    # Calculate moving averages for Reputation
    monthly_data['Reputation_6M_MA'] = monthly_data['Reputation'].rolling(window=6).mean()
    monthly_data['Reputation_3M_MA'] = monthly_data['Reputation'].rolling(window=3).mean()

    # Forecast the next 3 months for Reputation
    forecast_df = forecast_reputation(monthly_data)

    # Combine historical and forecasted data
    combined_data = pd.concat([monthly_data, forecast_df], ignore_index=True, sort=False)

    # Plot the data
    plot_reputation_time_series(combined_data, store_name=selected_stores)

    # Display the table below the chart
    display_reputation_table(monthly_data)
    
    


# Main application logic

def main():
    uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

    add_logo("logo_freatz.png")

    # Define menu options and corresponding icons
    menu_options = [
        "All Stores: 7 KPIs",
        "All Stores: All KPIs",
        "All Stores: Top-Down Analysis",
        "All Stores: Correlation Matrix",
        "All Stores: Category Impact",
        "All Stores: Customer Journey",
        "All Stores: What If (Impact Analysis)",
        "All Stores: Pivot Table",
        "All Stores: Reputation Index", 
        "All Stores: Plan of Action",  
        "Time-Series",
        "Ask Me Anything",
        "__________________",
        "Individual Store: 7 KPIs",
        "Individual Store: All KPIs",
        "Individual Store: Top-Down Analysis",
        "Individual Store: Correlation Matrix",
        "Individual Store: Category Impact",
        "Individual Store: Customer Journey",
        "Individual Store: What If (Impact Analysis)",
        "Individual Store: Pivot Table",
        "Individual Store: Reputation Index",
        "__________________",
        "Other Menus",
        "Past Predictions",
        "Diagnostic Information",
        "Cross-Validation",
    ]

    menu_icons = [
        "bi bi-bar-chart",          # All Stores: 7 KPIs
        "bi bi-bar-chart-fill",     # All Stores: All KPIs
        "bi bi-sliders",            # All Stores: Top-Down Analysis
        "bi bi-activity",           # All Stores: Correlation Matrix
        "bi bi-tools",              # All Stores: Category Impact
        "bi bi-map",                # All Stores: Customer Journey
        "bi bi-lightbulb",          # All Stores: What If (Impact Analysis)
        "bi bi-table",              # All Stores: Pivot Table
        "bi bi-clipboard-data",     # All Stores: Plan of Action
        "bi bi-star",               # All Stores: Reputation Index
        "bi bi-person-circle",      # Individual Store: 7 KPIs
        "bi bi-person-lines-fill",  # Individual Store: All KPIs
        "bi bi-person-check",       # Individual Store: Top-Down Analysis
        "bi bi-graph-up",           # Individual Store: Correlation Matrix
        "bi bi-bar-chart-steps",    # Individual Store: Category Impact
        "bi bi-geo-alt",            # Individual Store: Customer Journey
        "bi bi-puzzle",             # Individual Store: What If (Impact Analysis)
        "bi bi-tablet",             # Individual Store: Pivot Table
        "bi bi-check-circle",       # Cross-Validation
        "bi bi-info-circle",        # Diagnostic Information
        "bi bi-clock-history",      # Time-Series
        "bi bi-pie-chart-fill",     # Past Predictions
        "bi bi-question-circle",    # Ask Me Anything

    ]

    # Implement the option menu in the sidebar with custom styles
    with st.sidebar:
        selected = option_menu(
            menu_title=None,  # No title displayed
            options=menu_options,
            icons=menu_icons,
            menu_icon="cast",  # Sidebar icon
            default_index=0,
            orientation="vertical",
            styles={
                "container": {
                    "padding": "5px 0px",
                    "background-color": "#fafafa"
                },
                "icon": {
                    "color": "#FD575B",
                    "font-size": "18px"
                },
                "nav-link": {
                    "font-size": "16px",
                    "text-align": "left",
                    "margin":"0px",
                    "--hover-color": "#eee"
                },
                "nav-link-selected": {
                    "background-color": "#FD575B"
                },
            }
        )

    # Assign the selected option to 'page' variable
    page = selected

    if uploaded_file is not None:
        file_id, webViewLink = upload_file_to_drive(uploaded_file.name, uploaded_file)
        store_uploaded_file_link(uploaded_file.name, webViewLink)
        
        df = pd.read_excel(uploaded_file, sheet_name='Converted 0 and 1', engine='openpyxl', header=None)

        try:
            kpi_actions_upper = pd.read_excel(uploaded_file, sheet_name='KPIs_to_Actions-UpperManag', engine='openpyxl')
            kpi_actions_consumers = pd.read_excel(uploaded_file, sheet_name='KPIs_to_Actions-Consumers', engine='openpyxl')
            kpi_actions_freatz = pd.read_excel(uploaded_file, sheet_name='KPIs_to_Actions-Freatz', engine='openpyxl')


        except Exception as e:
            st.error(f"Error reading action sheets: {e}")
            st.stop()

        categories = df.iloc[0]
        step_numbers = df.iloc[1]
        moments = df.iloc[2]
        column_names = df.iloc[3]
        
        df.columns = column_names
        df = df.iloc[4:].reset_index(drop=True)

        categories.index = df.columns
        step_numbers.index = df.columns
        moments.index = df.columns
        column_names.index = df.columns

        est_name_col = 'establishment_name'
        targets = ['satisfaction', 'value_for_money', 'return_probability', 'nps']

        feature_columns = [col for col in df.columns if col not in exclude_columns + targets]

        categories = categories[feature_columns]
        step_numbers = step_numbers[feature_columns]
        moments = moments[feature_columns]
        column_names = column_names[feature_columns]

        features = df[feature_columns]
        features_converted = features.apply(pd.to_numeric, errors='coerce')
        features_converted = features_converted.select_dtypes(include=[np.number])

        if features_converted.empty:
            st.error("The features DataFrame is empty after selecting numeric columns. Please check your data.")
            st.stop()

        imputer = SimpleImputer(strategy='most_frequent')
        X_imputed = pd.DataFrame(imputer.fit_transform(features_converted), columns=features_converted.columns)

        y_imputed = df[targets].apply(pd.to_numeric, errors='coerce')
        y_imputed = y_imputed.fillna(y_imputed.mode().iloc[0])

        for target in targets:
            y_imputed[target] = y_imputed[target].astype(int)

        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X_imputed.columns)

        indicator_mapping = {
            'satisfaction': 'Wow Factor',
            'value_for_money': 'Perceived Value',
            'nps': 'NPS/Recommendation',
            'return_probability': 'Return Rate'
        }

        models, combined_scores = train_models(X_scaled, y_imputed, targets)

        # Define column_metadata AFTER preprocessing
        column_metadata = pd.DataFrame({
            'Column Name': column_names,
            'Category': categories,
            'Step Number': step_numbers,
            'Moment': moments
        })
        
        # Handle different pages
        if page in ["All Stores: 7 KPIs", "Individual Store: 7 KPIs"]:
            if "Individual" in page:
                st.title("Individual Store: 7 KPIs Analysis")
                selected_stores = select_stores(df, selector_key="individual_7_kpis_store_selector")
                if selected_stores:
                    st.write(f"**Analysis for selected store(s):** {', '.join(selected_stores)}")
                    df_filtered = filter_dataframe(df, selected_stores)
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

                    display_top_7_kpis(
                        X_scaled_filtered, 
                        y_imputed_filtered, 
                        models, 
                        combined_scores, 
                        scaler, 
                        targets, 
                        indicator_mapping, 
                        column_names, 
                        X_imputed_filtered
                    )
                else:
                    st.warning("Please select at least one store to proceed with the analysis.")
            else:
                st.title("All Stores: 7 KPIs Analysis")
                display_top_7_kpis(
                    X_scaled, 
                    y_imputed, 
                    models, 
                    combined_scores, 
                    scaler, 
                    targets, 
                    indicator_mapping, 
                    column_names, 
                    X_imputed
                )

        elif page in ["All Stores: All KPIs", "Individual Store: All KPIs"]:
            if "Individual" in page:
                st.title("Individual Store: All KPIs Analysis")
                selected_stores = select_stores(df, selector_key="individual_all_kpis_store_selector")
                if selected_stores:
                    st.write(f"**Analysis for selected store(s):** {', '.join(selected_stores)}")
                    df_filtered = filter_dataframe(df, selected_stores)
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

                    display_all_kpis(
                        X_scaled_filtered, 
                        y_imputed_filtered, 
                        models, 
                        combined_scores, 
                        scaler, 
                        targets, 
                        indicator_mapping, 
                        column_names, 
                        X_imputed_filtered
                    )
                else:
                    st.warning("Please select at least one store to proceed with the analysis.")
            else:
                st.title("All Stores: All KPIs Analysis")
                display_all_kpis(
                    X_scaled, 
                    y_imputed, 
                    models, 
                    combined_scores, 
                    scaler, 
                    targets, 
                    indicator_mapping, 
                    column_names, 
                    X_imputed
                )

        elif page in ["All Stores: Top-Down Analysis", "Individual Store: Top-Down Analysis"]:
            st.title("Top-Down Analysis")
            if "Individual" in page:
                selected_stores = select_stores(df, selector_key="top_down_store_selector")
                if selected_stores:
                    st.write(f"**Analysis for selected store(s):** {', '.join(selected_stores)}")
                    df_filtered = filter_dataframe(df, selected_stores)
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
                correlation_numeric_df = X_imputed.copy()
                display_correlation_matrix(df, correlation_numeric_df)
            else:
                display_individual_correlation_matrix(df, X_imputed, display_correlation_matrix)

        elif page in ["All Stores: Category Impact", "Individual Store: Category Impact"]:
            if "Individual" in page:
                st.title("Individual Store: Category Impact")
                selected_stores = select_stores(df, selector_key="category_impact_store_selector")
                if selected_stores:
                    st.write(f"**Analysis for selected store(s):** {', '.join(selected_stores)}")
                    df_filtered = filter_dataframe(df, selected_stores)
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

                    analyze_category_impact(
                        X_scaled_filtered, 
                        X_imputed_filtered, 
                        y_imputed_filtered, 
                        models, 
                        combined_scores, 
                        categories, 
                        column_names, 
                        scaler, 
                        targets, 
                        indicator_mapping
                    )
                else:
                    st.warning("Please select at least one store to proceed with the analysis.")
            else:
                st.title("All Stores: Category Impact")
                analyze_category_impact(
                    X_scaled, 
                    X_imputed, 
                    y_imputed, 
                    models, 
                    combined_scores, 
                    categories, 
                    column_names, 
                    scaler, 
                    targets, 
                    indicator_mapping
                )



        elif page == "All Stores: What If (Impact Analysis)":
            display_what_if_analysis(df, X_imputed, y_imputed, targets, indicator_mapping)

        elif page == "Individual Store: What If (Impact Analysis)":
            display_individual_what_if_analysis(df, X_imputed, y_imputed, targets, indicator_mapping)  

        elif page == "All Stores: Pivot Table":
            display_all_stores_pivot_table(df)

        elif page == "Individual Store: Pivot Table":
            display_individual_pivot_table(df)

        elif page == "Ask Me Anything":
            display_ask_me_anything(df)


        elif page in ["All Stores: Reputation Index"]:
            df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['month'].astype(str), errors='coerce')
            display_reputation_index(df)

        elif page == "Individual Store: Reputation Index":
            df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['month'].astype(str), errors='coerce')
            display_individual_store_reputation_index(df)


        elif page == "Cross-Validation":
            st.title("Cross-Validation Results")
            for target in targets:
                st.write(f"**{indicator_mapping[target]}:**")
                
                for model_name, model in models[target].items():
                    cv = 5
                    scores = cross_val_score(model, X_scaled, y_imputed[target], cv=cv, scoring='accuracy')
                    st.write(f"  - {model_name} Accuracy ({cv}-fold CV): {scores.mean():.2f}")
                
                combined_model = LinearRegression()
                cv = 5
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
                'KPI': column_names.values
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
            df_numeric_converted = df.apply(pd.to_numeric, errors='coerce')
            numeric_df = df_numeric_converted.select_dtypes(include=[np.number])
            if not numeric_df.empty and numeric_df.shape[1] > 0:
                st.write(numeric_df.describe())
            else:
                st.write("No numeric columns found in the dataset.")

            display_model_performance_comparison(X_scaled, y_imputed, models, targets, indicator_mapping)

        elif page == "Time-Series":
            display_time_series_analysis(df.copy(), targets, indicator_mapping)

        elif page == "Past Predictions":
            st.title("Past Predictions")
            display_past_predictions()

        elif page in ["All Stores: Customer Journey", "Individual Store: Customer Journey"]:
            # Add Date Selection Sidebar
            st.sidebar.header("Customer Journey Date Selection")
            try:
                df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['month'].astype(str), errors='coerce')
                
            except Exception as e:
                st.error(f"Error parsing 'month' and 'Year' columns into datetime: {e}")
                return

            min_date = df['Date'].min()
            max_date = df['Date'].max()

            start_date = st.sidebar.date_input("Start date", min_date, min_value=min_date, max_value=max_date, key="customer_journey_start_date")
            end_date = st.sidebar.date_input("End date", max_date, min_value=min_date, max_value=max_date, key="customer_journey_end_date")

            if start_date > end_date:
                st.error("Start date must be before or equal to End date.")
                return

            date_range = (start_date, end_date)

            if page == "All Stores: Customer Journey":
                display_all_stores_customer_journey(df, column_metadata, date_range)
            else:
                # Individual Store: Customer Journey
                st.sidebar.header("Select Store for Customer Journey")
                unique_stores = df['establishment_name'].unique()
                selected_store = st.sidebar.selectbox('Select store:', unique_stores, key="customer_journey_store_selector")
                
                display_individual_store_customer_journey(df, column_metadata, date_range)

        elif page == "All Stores: Plan of Action":  # <-- New Menu Handling
            display_plan_of_action(
                df=df,
                X_imputed=X_imputed,
                y_imputed=y_imputed,
                models=models,
                combined_scores=combined_scores,
                scaler=scaler,
                targets=targets,
                indicator_mapping=indicator_mapping,
                kpi_actions_upper=kpi_actions_upper,
                kpi_actions_consumers=kpi_actions_consumers,
                kpi_actions_freatz=kpi_actions_freatz,
                column_metadata=column_metadata
            )


            # Call the test function - REMOVE FROM THE CODE LATER ON - JUST VALIDATING OPENAI KEY AND API CONNECTION
            test_openai_connection()



    else:
        st.write("Please upload an Excel file to begin the analysis.")

# Make sure these functions are defined earlier in your script:

def plot_customer_journey(df_data, column_metadata):
    # Calculate percentage of 1s for each KPI, ignoring blank values
    kpi_percentages = df_data.apply(lambda col: col.dropna().mean() * 100)
    kpi_percentages = kpi_percentages.reset_index()
    kpi_percentages.columns = ['Column Name', 'Percentage']

    # Combine metadata with KPI percentages
    df_combined = column_metadata.merge(kpi_percentages, on='Column Name', how='left')

    # Debugging: Check if 'Moment' exists
    st.write("### Combined Data for Customer Journey:")
    st.write(df_combined.head())

    # Verify 'Moment' column exists
    if 'Moment' not in df_combined.columns:
        st.error("The 'Moment' column is missing after merging. Please check your column_metadata.")
        return

    # Order moments by Step Number
    df_combined['Step Number'] = pd.to_numeric(df_combined['Step Number'], errors='coerce')
    df_combined = df_combined.sort_values('Step Number')

    # Plotting
    fig_height = max(20, 4 * len(df_combined['Moment'].unique()))
    fig_width = 20
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.05)

    for line_num, moment in enumerate(reversed(df_combined['Moment'].unique())):
        moment_data = df_combined[df_combined['Moment'] == moment]
        scores = moment_data['Percentage'].values
        labels = moment_data['Column Name'].values
        positions = range(len(scores))

        y_position = line_num * 2

        ax.plot(positions, [y_position] * len(scores), linestyle='-', color='gray', linewidth=3, alpha=0.7)

        for i, (pos, score, label) in enumerate(zip(positions, scores, labels)):
            color = plt.cm.viridis(score / 100)  # Color gradient based on score
            ax.scatter(pos, y_position, color=color, s=300, zorder=5, marker='o')
            ax.text(pos, y_position + 0.4, f'{score:.0f}%', fontsize=20, ha='center', va='bottom')
            ax.text(pos, y_position - 0.4, textwrap.fill(label, width=20), fontsize=13, ha='center', va='top')

        avg_score = moment_data['Percentage'].mean()
        ax.text(-0.5, y_position, f"{moment} ({avg_score:.0f}%)", fontsize=16, ha='right', va='center', fontweight='bold')

    ax.set_title("Customer Journey Flow", fontsize=18, fontweight='bold', pad=20)
    ax.set_yticks([])
    ax.set_ylim(-1, len(df_combined['Moment'].unique()) * 2)

    legend_labels = ['High Friction (< 60%)', 'Medium Friction (60%-75%)', 'Goal Achieved (> 75%)']
    legend_colors = [plt.cm.viridis(0.25), plt.cm.viridis(0.6), plt.cm.viridis(0.9)]
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=10) for c in legend_colors]
    ax.legend(handles, legend_labels, title="Score Breakdown", loc='upper left', bbox_to_anchor=(1, 1))

    st.pyplot(fig)

    return df_combined

def text_representation(df_combined):
    st.header("Customer Journey Flow - Text Representation")
    
    unique_moments = df_combined['Moment'].unique()
    
    for moment in unique_moments:
        moment_data = df_combined[df_combined['Moment'] == moment]
        avg_score = moment_data['Percentage'].mean()
        
        st.subheader(f"{moment} (Average: {avg_score:.0f}%)")
        
        for _, row in moment_data.iterrows():
            kpi = row['Column Name']
            score = row['Percentage']
            if score < 60:
                dot_color = "#FFCCCB"  # Light red
            elif 60 <= score < 75:
                dot_color = "#FFFFA1"  # Light yellow
            else:
                dot_color = "#90EE90"  # Light green
            
            st.markdown(f'<span style="color:{dot_color};">●</span> {kpi}: {score:.0f}%', unsafe_allow_html=True)
        
        st.write("---")

def get_worst_moment(df_combined):
    moment_averages = df_combined.groupby('Moment')['Percentage'].mean()
    worst_moment = moment_averages.idxmin()
    worst_score = moment_averages.min()
    return worst_moment, worst_score

def get_okr_summary(df_combined):
    okrs = ['satisfaction', 'value_for_money', 'return_probability', 'nps']
    okr_summary = df_combined[df_combined['Column Name'].isin(okrs)].set_index('Column Name')['Percentage']
    return okr_summary

if __name__ == "__main__":
    main()
