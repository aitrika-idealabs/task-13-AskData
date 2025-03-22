import streamlit as st
import pandas as pd
import numpy as np

def create_interactive_visualization(df):
    """Create an interactive visualization with selectable axes and chart types"""
    st.write("### Data Visualization")
    
    # Create columns for the controls
    col1, col2, col3 = st.columns([1, 1, 2])
    
    # Get all column names for selection
    all_columns = df.columns.tolist()
    
    # X-axis selection
    with col1:
        x_axis = st.selectbox(
            "X-Axis",
            options=all_columns,
            index=0 if all_columns else None,
            key="x_axis_select"
        )
        st.session_state["selected_x_axis"] = x_axis
    
    # Y-axis selection    
    with col2:
        # Get numeric columns for y-axis
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        y_axis = st.selectbox(
            "Y-Axis",
            options=numeric_columns if numeric_columns else all_columns,
            index=1 if len(numeric_columns) > 1 else (0 if numeric_columns else None),
            key="y_axis_select"
        )
        st.session_state["selected_y_axis"] = y_axis
    
    # Chart type selection
    with col3:
        chart_type = st.radio(
            "Chart Type",
            options=["Bar", "Line", "Histogram"],
            horizontal=True,
            key="chart_type_select"
        )
    
    # Create visualization based on selections
    if x_axis and y_axis:
        # Prepare data
        chart_data = df.copy()
        
        try:
            # Create appropriate chart based on selection
            if chart_type == "Bar":
                st.bar_chart(
                    chart_data,
                    x=x_axis,
                    y=y_axis
                )
            elif chart_type == "Line":
                st.line_chart(
                    chart_data,
                    x=x_axis,
                    y=y_axis
                )
            elif chart_type == "Histogram":
                # Create a histogram of the selected y-axis column
                st.write(f"Histogram of {y_axis}")
                
                # Check if the selected column is numeric
                if pd.api.types.is_numeric_dtype(chart_data[y_axis]):
                    # For histograms, we only need one column (y_axis)
                    # Create bins for the histogram
                    hist_values = chart_data[y_axis].dropna()
                    
                    # Create a DataFrame for the histogram
                    hist_data = pd.DataFrame()
                    # Calculate the bins and frequencies
                    hist, bin_edges = np.histogram(hist_values, bins=10)
                    
                    # Create labels for the bins
                    bin_labels = [f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}" for i in range(len(bin_edges)-1)]
                    
                    # Create a DataFrame from the histogram data
                    hist_df = pd.DataFrame({
                        'bin': bin_labels,
                        'frequency': hist
                    })
                    
                    # Display the histogram using a bar chart
                    st.bar_chart(
                        hist_df,
                        x='bin',
                        y='frequency'
                    )
                else:
                    st.error(f"Cannot create histogram: '{y_axis}' is not a numeric column")
                    st.info("Please select a numeric column for the y-axis when using a histogram")
                
        except Exception as e:
            st.error(f"Could not create {chart_type.lower()} chart: {e}")
            st.info("Try selecting different columns or chart types")