import streamlit as st
import pandas as pd
import uuid
import os
import io
import PyPDF2
import google.generativeai as genai
import numpy as np

# Import modules
from file_processing import process_csv_file, process_excel_file, extract_text_from_pdf
from visualization import create_interactive_visualization
from gemini_handler import generate_gemini_response

# Set Gemini API Key
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# Initialize session state
if "context" not in st.session_state:
    st.session_state["context"] = ""
if "data" not in st.session_state:
    st.session_state["data"] = None
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "selected_x_axis" not in st.session_state:
    st.session_state["selected_x_axis"] = None
if "selected_y_axis" not in st.session_state:
    st.session_state["selected_y_axis"] = None
if "df_json" not in st.session_state:
    st.session_state["df_json"] = None
if "file_type" not in st.session_state:
    st.session_state["file_type"] = None

def clear_chat_history():
    st.session_state["chat_history"] = []
    st.rerun()

# Streamlit App
st.title("Ask Data - Chat with Your Files")

# File Upload
uploaded_file = st.file_uploader("Upload your file", type=["csv", "xlsx", "pdf"])

if uploaded_file:
    # Load Data based on file type
    file_type = uploaded_file.name.split('.')[-1].lower()
    st.session_state["file_type"] = file_type
    
    if file_type == "csv":
        # Process CSV file
        df, context = process_csv_file(uploaded_file)
        st.session_state["data"] = df
        st.session_state["context"] = context
        
        # Display data preview
        st.write("### Data Preview")
        st.dataframe(st.session_state["data"].head())
        
        # Show indicator that full data is captured
        st.success(f"✅ Full data extracted ({len(st.session_state['data'])} rows)")
        
        # Add checkbox to toggle visualization
        show_viz = st.checkbox("Show Visualization", value=True, key="show_visualization")
        
        # Create interactive visualization only if checkbox is checked
        if show_viz:
            create_interactive_visualization(df)
        
    elif file_type == "xlsx":
        # Process Excel file
        df, context = process_excel_file(uploaded_file)
        st.session_state["data"] = df
        st.session_state["context"] = context
        
        # Display data preview
        st.write("### Data Preview")
        st.dataframe(st.session_state["data"].head())
        
        # Show indicator that full data is captured
        st.success(f"✅ Full data extracted ({len(st.session_state['data'])} rows)")
        
        # Add checkbox to toggle visualization
        show_viz = st.checkbox("Show Visualization", value=True, key="show_visualization")
        
        # Create interactive visualization only if checkbox is checked
        if show_viz:
            create_interactive_visualization(df)
        
    elif file_type == "pdf":
        # Extract PDF text
        pdf_text = extract_text_from_pdf(uploaded_file)
        st.session_state["context"] = f"PDF Content:\n{pdf_text}"
        
        # Show success message without preview for PDF
        st.success("✅ PDF content extracted successfully")

# User Query with Submit Button
col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_input("Ask a question about your data:")
with col2:
    # Add some vertical spacing to align with the text input
    st.write("")  
    submit_button = st.button("Ask", type="primary")

if submit_button and query and st.session_state["context"]:
    # Add query to history
    st.session_state["chat_history"].append({"role": "user", "content": query})
    
    # Show a spinner while waiting for Gemini's response
    with st.spinner("Thinking..."):
        # Generate response using Gemini
        try:
            answer = generate_gemini_response(query, st.session_state)
            
            # Add response to history
            st.session_state["chat_history"].append({"role": "assistant", "content": answer})
            
            # Display answer
            st.write("### Answer:")
            st.write(answer)
            
            # Auto Visualization for tabular data based on query
            if st.session_state["data"] is not None and ("visualize" in query.lower() or "chart" in query.lower() or "plot" in query.lower() or "graph" in query.lower()):
                df = st.session_state["data"]
                st.write("### Query-Based Visualization:")
                
                # Use the previously selected axes if available
                x_axis = st.session_state["selected_x_axis"] if st.session_state["selected_x_axis"] else df.columns[0]
                
                # Try to find a numeric column for y-axis if not already selected
                if not st.session_state["selected_y_axis"]:
                    numeric_cols = df.select_dtypes(include=['number']).columns
                    y_axis = numeric_cols[0] if len(numeric_cols) > 0 else df.columns[1] if len(df.columns) > 1 else df.columns[0]
                else:
                    y_axis = st.session_state["selected_y_axis"]
                
                # Decide on visualization type based on query
                if "bar" in query.lower():
                    st.bar_chart(df, x=x_axis, y=y_axis)
                elif "line" in query.lower() or "trend" in query.lower():
                    st.line_chart(df, x=x_axis, y=y_axis)
                elif "histogram" in query.lower() or "distribution" in query.lower():
                    # Check if y_axis is numeric
                    if pd.api.types.is_numeric_dtype(df[y_axis]):
                        # Create histogram
                        hist_values = df[y_axis].dropna()
                        hist, bin_edges = np.histogram(hist_values, bins=10)
                        bin_labels = [f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}" for i in range(len(bin_edges)-1)]
                        hist_df = pd.DataFrame({
                            'bin': bin_labels,
                            'frequency': hist
                        })
                        st.bar_chart(hist_df, x='bin', y='frequency')
                    else:
                        st.error(f"Cannot create histogram: '{y_axis}' is not a numeric column")
                else:
                    # Default to bar chart
                    st.bar_chart(df, x=x_axis, y=y_axis)
        
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            st.write("Please check your Gemini API key or try again later.")

# Display chat history
if st.session_state["chat_history"]:
    st.write("### Chat History")
    for message in st.session_state["chat_history"]:
        role = message["role"]
        content = message["content"]
        
        if role == "user":
            st.write(f"**You**: {content}")
        else:
            st.write(f"**Assistant**: {content}")
    
    # Add button to clear chat history
    if st.button("Clear Chat History"):
        clear_chat_history()
