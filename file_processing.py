import pandas as pd
import json
import PyPDF2
import numpy as np

def process_csv_file(uploaded_file):
    """
    Process CSV file and prepare both dataframe and context for analysis
    
    Returns:
        tuple: (dataframe, context_string)
    """
    # Read CSV file
    df = pd.read_csv(uploaded_file)
    
    # Create a JSON serializable version for context
    # First handle non-serializable data types
    df_json = df.copy()
    
    # Handle NaN values and other non-serializable data types
    for col in df_json.columns:
        if df_json[col].dtype == 'datetime64[ns]':
            df_json[col] = df_json[col].astype(str)
        if df_json[col].isna().any():
            df_json[col] = df_json[col].fillna('null')
    
    # Prepare statistical context
    stats_context = generate_statistical_context(df)
    
    # Convert to JSON
    df_json_str = df_json.to_json(orient='records', date_format='iso')
    
    # Combine context information
    context = f"""
    CSV Data Statistical Summary:
    {stats_context}
    
    Full Data (JSON format):
    {df_json_str}
    """
    
    return df, context

def process_excel_file(uploaded_file):
    """
    Process Excel file and prepare both dataframe and context for analysis
    
    Returns:
        tuple: (dataframe, context_string)
    """
    # Read Excel file
    df = pd.read_excel(uploaded_file)
    
    # Create a JSON serializable version for context
    # First handle non-serializable data types
    df_json = df.copy()
    
    # Handle NaN values and other non-serializable data types
    for col in df_json.columns:
        if df_json[col].dtype == 'datetime64[ns]':
            df_json[col] = df_json[col].astype(str)
        if df_json[col].isna().any():
            df_json[col] = df_json[col].fillna('null')
    
    # Prepare statistical context
    stats_context = generate_statistical_context(df)
    
    # Convert to JSON
    df_json_str = df_json.to_json(orient='records', date_format='iso')
    
    # Combine context information
    context = f"""
    Excel Data Statistical Summary:
    {stats_context}
    
    Full Data (JSON format):
    {df_json_str}
    """
    
    return df, context

def extract_text_from_pdf(pdf_file):
    """
    Extract text content from a PDF file
    
    Returns:
        str: Extracted text from PDF
    """
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def generate_statistical_context(df):
    """
    Generate comprehensive statistical context from a dataframe
    
    Returns:
        str: Statistical summary of the dataframe
    """
    # Column information
    cols_info = f"Columns: {', '.join(df.columns)}"
    
    # Basic info
    basic_info = f"Rows: {len(df)}, Columns: {len(df.columns)}"
    
    # Data types
    dtypes_info = "Column Data Types:\n"
    for col, dtype in df.dtypes.items():
        dtypes_info += f"- {col}: {dtype}\n"
    
    # Numerical statistics
    try:
        numerical_stats = "Numerical Statistics:\n"
        num_df = df.select_dtypes(include=['number'])
        if not num_df.empty:
            stats = num_df.describe().to_string()
            numerical_stats += stats + "\n"
        else:
            numerical_stats += "No numerical columns found.\n"
    except Exception as e:
        numerical_stats = f"Could not generate numerical statistics: {str(e)}\n"
    
    # Categorical statistics
    try:
        cat_stats = "Categorical Statistics:\n"
        cat_df = df.select_dtypes(exclude=['number'])
        if not cat_df.empty:
            for col in cat_df.columns:
                value_counts = df[col].value_counts().head(10)  # Top 10 most common values
                cat_stats += f"- {col}: {value_counts.to_dict()}\n"
        else:
            cat_stats += "No categorical columns found.\n"
    except Exception as e:
        cat_stats = f"Could not generate categorical statistics: {str(e)}\n"
    
    # Missing values
    missing_values = "Missing Values:\n"
    for col in df.columns:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            missing_values += f"- {col}: {missing_count} missing values ({(missing_count/len(df))*100:.2f}%)\n"
    
    # Combine all statistics
    stats_context = f"{cols_info}\n\n{basic_info}\n\n{dtypes_info}\n{numerical_stats}\n{cat_stats}\n{missing_values}"
    
    return stats_context