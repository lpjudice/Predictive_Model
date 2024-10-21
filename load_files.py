from googleapiclient.discovery import build

def list_service_account_files():
    # Build the Google Drive API service with the service account credentials
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
