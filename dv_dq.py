import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Remove the function wrapper and keep the main code
st.markdown("""
Please upload an Excel file (.xlsx) that contains the following columns: 
- **Voltage(V)** starts from Cell **A1**
- **Capacity(Ah)** from Cell **B1**. 

Please match the columns name as : Voltage(V) and Capacity(Ah)
""")
st.markdown("""This is not done by Interpolation method because interpolation method takes 
only Max and Min Voltage from original data points and estimates values based on the available data""")
st.markdown("""**Please upload Charge and Discharge data in separate excel files**""")
st.markdown("""Please let me know if any error occurs. **Thank You !**""")

def gaussian_kernel(size, sigma):
    x = np.arange(-size // 2 + 1, size // 2 + 1)
    g = np.exp(-(x ** 2 / (2 * sigma ** 2)))
    return g / np.sum(g)

def gaussian_smooth(curve, kernel_size, sigma):
    kernel = gaussian_kernel(kernel_size, sigma)
    smoothed = np.convolve(curve, kernel, mode='same')
    smoothed /= np.sum(kernel)
    return smoothed

def process_data_dvdq(df, matched_array):
    capacity_array = df['Capacity(Ah)'].values
    matching_indices = []
    i = 0
    while i < len(capacity_array):
        diffs = np.round(capacity_array[i+1:] - capacity_array[i], 4)
        matches = np.isin(diffs, matched_array)
        if np.any(matches):
            matching_indices.append(i)
            i += np.argmax(matches) + 1
        else:
            i += 1

    new_df = df.iloc[matching_indices]
    new_df['dvdq'] = (new_df['Voltage(V)'].diff() / new_df['Capacity(Ah)'].diff()).shift(-1)
    new_df['Gaussian_smooth'] = gaussian_smooth(new_df['dvdq'], 3, 1)
    return new_df

# Updated charge and discharge matched arrays
charge_matched_array = np.array([0.01, 0.02])  # Example for charge
discharge_matched_array = np.array([-0.01, -0.02])  # Example for discharge

uploaded_file = st.file_uploader("Upload an excel file", type=['xlsx'])

if uploaded_file is not None:
    data = pd.read_excel(uploaded_file)
    option = st.text_input("Enter 'Charge' if data is of Charging or 'Discharge' if data is of Discharge:")
    if option.lower() == 'charge':
        new_df = process_data_dvdq(data, charge_matched_array)
        st.dataframe(new_df)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=new_df['Capacity(Ah)'], y=new_df['dvdq'], mode='lines', name='dvdq'))
        fig.add_trace(go.Scatter(x=new_df['Capacity(Ah)'], y=new_df['Gaussian_smooth'], mode='lines', name='Gaussian_smooth'))
        fig.update_layout(title='Capacity(Ah) vs dvdq and Gaussian_smooth', xaxis_title='Capacity(Ah)', yaxis_title='dvdq and Gaussian_smooth')
        st.plotly_chart(fig)
        csv_data = new_df.to_csv(index=False)
        st.markdown("### Download Filtered CSV File")
        st.download_button("Download", data=csv_data, file_name="filtered_data.csv", mime='text/csv')
    elif option.lower() == 'discharge':
        new_df = process_data_dvdq(data, discharge_matched_array)
        st.dataframe(new_df)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=new_df['Capacity(Ah)'], y=new_df['dvdq'], mode='lines', name='dvdq'))
        fig.add_trace(go.Scatter(x=new_df['Capacity(Ah)'], y=new_df['Gaussian_smooth'], mode='lines', name='Gaussian_smooth'))
        fig.update_layout(title='Capacity(Ah) vs dvdq and Gaussian_smooth', xaxis_title='Capacity(Ah)', yaxis_title='dvdq and Gaussian_smooth')
        st.plotly_chart(fig)
        csv_data = new_df.to_csv(index=False)
        st.markdown("### Download Filtered CSV File")
        st.download_button("Download", data=csv_data, file_name="filtered_data.csv", mime='text/csv')
    else:
        st.write("Please enter either 'Charge' or 'Discharge'.")
else:
    st.write("Please upload a file.")
