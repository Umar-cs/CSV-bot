#import req libraries
import streamlit as st
import pandas as pd
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
import matplotlib
#TKAgg works fine other wise Agg on second checked on 7 sep 2023
matplotlib.use('TKAgg')

# Streamlit sidebar to input the OpenAI API token
st.sidebar.title("OpenAI API Configuration")
api_token = st.sidebar.text_input("Enter your OpenAI API token:" , type="password")

# Check if API token is provided
if not api_token:
    st.warning("Please enter your OpenAI API token in the sidebar.")
else:
    st.title("csv bot ka title")

    # File upload section
    uploaded_file = st.file_uploader("Upload a CSV file for analysis", type=['csv'])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(df.head(10))

        # User prompt input
        prompt = st.text_area("Enter your prompt:")

        # Create LLM and PandasAI objects if API token is provided
        openai_llm = OpenAI(api_token=api_token)
        pandas_ai = PandasAI(openai_llm)

        # Generate output based on the prompt
        if st.button("Generate"):
            if prompt:
                # Generate response
                with st.spinner("Generating response..."):
                    st.write(pandas_ai.run(df, prompt))
            else:
                st.warning("Please enter a prompt.")
