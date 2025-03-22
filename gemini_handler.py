import google.generativeai as genai
import json
import pandas as pd
import numpy as np

def generate_gemini_response(query, session_state):
    """
    Generate a response from Gemini based on the query and session state
    
    Args:
        query (str): User's query
        session_state (dict): Streamlit session state containing context and data
    
    Returns:
        str: Response from Gemini
    """
    # Check if we're dealing with tabular data
    is_tabular = session_state["file_type"] in ["csv", "xlsx"]
    
    if is_tabular:
        # For tabular data, let's check if this is a numerical/analytical question
        numerical_keywords = [
            "average", "mean", "median", "mode", "sum", "total", "calculate", 
            "maximum", "minimum", "max", "min", "count", "how many", 
            "percentage", "ratio", "proportion", "standard deviation",
            "variance", "correlation", "trend", "growth", "stats", "statistics"
        ]
        
        is_numerical_question = any(keyword in query.lower() for keyword in numerical_keywords)
        
        if is_numerical_question:
            # For numerical questions, add additional context to help Gemini analyze the data
            prompt = create_numerical_analysis_prompt(query, session_state)
        else:
            # For general questions about tabular data
            prompt = create_general_tabular_prompt(query, session_state)
    else:
        # For PDF files, use a more general prompt
        prompt = create_text_analysis_prompt(query, session_state)
    
    # Generate response using Gemini
    model = genai.GenerativeModel('gemini-2.0-flash')
    response = model.generate_content(prompt)
    
    return response.text

def create_numerical_analysis_prompt(query, session_state):
    """Create a specialized prompt for numerical analysis questions"""
    # Get the dataframe for analysis examples
    df = session_state["data"]
    
    # Generate some example calculations to help Gemini understand the data
    examples = generate_calculation_examples(df)
    
    prompt = f"""
    You are a data analyst assistant specializing in numerical analysis. Answer the question using the provided data and context.
    
    The data comes from a {session_state["file_type"].upper()} file with {len(df)} rows and {len(df.columns)} columns.
    
    Here are some sample calculations to help you understand the data structure:
    {examples}
    
    The statistical summary of the data:
    {session_state["context"]}
    
    Question: {query}
    
    Provide a detailed analytical answer with precise numerical values when answering questions about calculations, statistics, or numerical trends.
    If performing calculations, show your reasoning clearly.
    """
    
    return prompt

def create_general_tabular_prompt(query, session_state):
    """Create a general prompt for tabular data questions"""
    prompt = f"""
    You are a data analyst assistant. Answer the question using the following context.
    This context contains the COMPLETE data from the user's {session_state["file_type"].upper()} file:
    
    {session_state["context"]}
    
    Question: {query}
    
    Provide a clear and concise answer based only on the information in the context.
    Don't limit your analysis to what might be shown in visualizations - use ALL the data in the context.
    If the answer cannot be determined from the context, say so.
    """
    
    return prompt

def create_text_analysis_prompt(query, session_state):
    """Create a prompt for text (PDF) analysis"""
    prompt = f"""
    You are a document analysis assistant. Answer the question using the following document content:
    
    {session_state["context"]}
    
    Question: {query}
    
    Provide a clear and concise answer based only on the information in the document.
    If the answer cannot be determined from the document content, say so.
    """
    
    return prompt

def generate_calculation_examples(df):
    """
    Generate example calculations on the dataframe to help Gemini understand the data
    
    Args:
        df (pandas.DataFrame): The dataframe to analyze
    
    Returns:
        str: Example calculations on the data
    """
    examples = []
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if numeric_cols:
        # Example 1: Basic statistics on a numeric column
        col = numeric_cols[0]
        examples.append(f"Column '{col}' statistics:")
        examples.append(f"- Mean: {df[col].mean()}")
        examples.append(f"- Median: {df[col].median()}")
        examples.append(f"- Min: {df[col].min()}")
        examples.append(f"- Max: {df[col].max()}")
        
        # Example 2: If we have at least 2 numeric columns, show correlation
        if len(numeric_cols) >= 2:
            col1, col2 = numeric_cols[0], numeric_cols[1]
            correlation = df[col1].corr(df[col2])
            examples.append(f"\nCorrelation between '{col1}' and '{col2}': {correlation:.4f}")
    
    # Example 3: Value counts for a categorical column (if exists)
    non_numeric_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    if non_numeric_cols:
        cat_col = non_numeric_cols[0]
        top_values = df[cat_col].value_counts().head(3).to_dict()
        examples.append(f"\nTop values in '{cat_col}':")
        for val, count in top_values.items():
            examples.append(f"- {val}: {count} occurrences")
    
    # Example 4: Group by example if we have categorical and numeric columns
    if non_numeric_cols and numeric_cols:
        group_col = non_numeric_cols[0]
        value_col = numeric_cols[0]
        grouped = df.groupby(group_col)[value_col].mean().head(3).to_dict()
        examples.append(f"\nAverage '{value_col}' grouped by '{group_col}' (first 3 groups):")
        for group, avg in grouped.items():
            examples.append(f"- {group}: {avg:.4f}")
    
    return "\n".join(examples)