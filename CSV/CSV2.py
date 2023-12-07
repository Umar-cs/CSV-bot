import streamlit as st 
import pandas as pd
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
import matplotlib.pyplot as plt

# Create an LLM by instantiating OpenAI object and passing API token
llm = OpenAI(api_token="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")

# Create PandasAI object, passing the LLM
pandas_ai = PandasAI(llm)
st.title("Prompt-driven data analysis with PandasAI")

uploaded_file = st.file_uploader("Upload a CSV file for analysis", type=['csv'])

def display_matplotlib_figure(fig):
    """
    Display a Matplotlib figure using st.pyplot().

    Parameters:
        fig (matplotlib.figure.Figure): Matplotlib figure to display.
    """
    st.pyplot(fig)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df.head(3))

    # New code below...
    prompt = st.text_area("Enter your prompt:")

    # Generate output
    if st.button("Generate"):
        if prompt:
            # Call pandas_ai.run(), passing the dataframe and prompt
            with st.spinner("Generating response..."):
                response = pandas_ai.run(df, prompt)
                st.write(response)  # Display text response

            # Example: Generate and display a Matplotlib plot
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3, 4, 5], [10, 20, 30, 40, 50])  # Example plot

            # Display the Matplotlib plot within Streamlit
            display_matplotlib_figure(fig)
        else:
            st.warning("Please enter a prompt.")
