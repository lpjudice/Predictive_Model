import os
import json
import base64
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
import streamlit


def authenticate_google_drive():
    """Authenticate and return the credentials for the Google Drive API."""
    
    # Set the scopes your app requires
    SCOPES = ['https://www.googleapis.com/auth/drive.file']
    
    # Get the base64-encoded service account key from the environment variable
    service_account_base64 = os.getenv('GOOGLE_CREDENTIALS')
    
    if service_account_base64:
        # Decode the base64-encoded JSON key
        service_account_json = base64.b64decode(service_account_base64).decode('utf-8')
        service_account_info = json.loads(service_account_json)
        
        # Use the service account key to create credentials
        creds = Credentials.from_service_account_info(service_account_info, scopes=SCOPES)
    else:
        raise EnvironmentError("GOOGLE_CREDENTIALS environment variable not set.")
    
    return creds


def list_service_account_files():
    """List files uploaded by the service account"""
    # Build the Google Drive API service using the service account credentials
    drive_service = build('drive', 'v3', credentials=authenticate_google_drive())
    
    # List files owned by the service account
    results = drive_service.files().list(
        pageSize=10, fields="nextPageToken, files(id, name)"
    ).execute()
    
    files = results.get('files', [])
    
    if not files:
        print('No files found.')
    else:
        print('Files:')
        for file in files:
            print(f"{file['name']} (ID: {file['id']})")

# Call the function to list files
list_service_account_files()

def display_files():
    """Display the list of files in Streamlit."""
    st.title('List of Files in Service Account')
    
    files = list_service_account_files()
    
    if not files:
        st.write('No files found.')
    else:
        st.write('Files found:')
        for file in files:
            st.write(f"{file['name']} (ID: {file['id']})")

# Call the display function to render the files in Streamlit
display_files()
