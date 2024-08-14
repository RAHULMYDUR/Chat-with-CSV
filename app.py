import os
import streamlit as st
import logging
import google.generativeai as genai
import pandas as pd
import traceback
import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up Google Cloud credentials
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("Google API Key is missing. Please set it in the .env file.")
    st.stop()
os.environ["GOOGLE_API_KEY"] = api_key

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_text_from_csv(uploaded_file):
    # Function to extract text from the uploaded CSV
    try:
        df = pd.read_csv(uploaded_file, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(uploaded_file, encoding='latin1')
        except Exception as e:
            logger.error(f"Error reading CSV file: {e}")
            st.error(f"An error occurred while reading the CSV file: {str(e)}")
            return None
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
        st.error(f"An error occurred while reading the CSV file: {str(e)}")
        return None

    df.columns = df.columns.str.strip()  # Clean up column names
    date_columns = [col for col in df.columns if 'date' in col.lower()]
    df[date_columns] = df[date_columns].apply(pd.to_datetime, errors='coerce')
    
    # Identify and convert numeric columns
    numeric_columns = [col for col in df.columns if df[col].dtype == 'object' and df[col].apply(lambda x: isinstance(x, str) and x.isnumeric()).all()]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Identify and convert boolean columns
    boolean_columns = [col for col in df.columns if df[col].apply(lambda x: isinstance(x, str) and x.lower() in ['true', 'false', '1', '0']).all()]
    for col in boolean_columns:
        df[col] = df[col].astype(bool)
    
    # Convert integer-like floats to integers
    integer_columns = [col for col in df.columns if df[col].dtype == 'float64' and df[col].dropna().apply(lambda x: x.is_integer()).all()]
    for col in integer_columns:
        df[col] = df[col].astype('Int64')

    return df

def clean_code(code):
    """Cleans generated code by removing markdown and stripping whitespace."""
    code = code.replace('```python', '').replace('```', '').strip()
    return code

def generate_text(prompt):
    """Generates text using the Gemini LLM."""
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    model = genai.GenerativeModel("gemini-1.5-pro-latest")
    response = model.generate_content(prompt)
    return response.text

def generate_code(user_query, df):
    # Generate the appropriate Python code snippet needed to answer the user's question.
    schema_info = df.dtypes.to_string()
    head_info = df.head().to_string(index=False)
    tail_info = df.tail().to_string(index=False)

    prompt = f'''
You are a chatbot that generates Python code snippets to perform specific operations on a DataFrame.

Dataset Schema:
{schema_info}

First few rows of the DataFrame:
{head_info}

User Question: "{user_query}"

Instructions:
- Identify the most relevant columns in the DataFrame that can help answer the question.
- Use logical, statistical, or computational methods as appropriate.
- Ensure the result of any operation is assigned to the variable 'result'.
- Handle different types of queries, such as filtering, aggregating, or probability estimation.
- Do not generate any explanation, generate only Python code.
- Preserve the case sensitivity of column names and values in your code.
- Always use 'df' as the DataFrame variable in your code.
- If a question cannot be answered with the available information or no relevant data is found, respond with "I apologize, I cannot answer that within my current scope."
- While generating graphs related code, keep in mind that it will be displayed in Streamlit, generate accordingly.
- Whatever the user asks, the question-related column only has to display in the output.

Generate the appropriate Python code snippet needed to answer the user's question.
'''

    generated_code = generate_text(prompt)
    cleaned_code = clean_code(generated_code)
    
    return cleaned_code

def execute_generated_code(generated_code, df):
    """Executes the generated Python code in a controlled environment."""
    if not isinstance(generated_code, str) or "I apologize" in generated_code or generated_code.strip() == "":
        st.error("The generated code is not valid. Please ask another question.")
        return None

    local_vars = {'df': df}

    try:
        logger.info(f"Executing generated code:\n{generated_code}")
        exec(generated_code, {}, local_vars)
        result = local_vars.get('result', None)

        if isinstance(result, list):
            if len(result) > 0 and isinstance(result[0], (int, float, str)):
                return pd.DataFrame({'Result': result})
            else:
                return pd.DataFrame(result)
        elif isinstance(result, (int, float, str, np.int64)):
            return pd.DataFrame({'Result': [result]})
        elif isinstance(result, pd.Series):
            return pd.DataFrame(result)
        elif isinstance(result, pd.DataFrame):
            return result
        else:
            st.error(f"Unexpected result type: {type(result)}")
            return None
    except KeyError as e:
        logger.error(f"KeyError: {e} - The generated code might be trying to access a column that doesn't exist.")
        st.error(f"A KeyError occurred: {str(e)}. Please check the generated code and column names.")
        return None
    except Exception as e:
        logger.error(f"Error executing generated code: {e}")
        st.error(f"An error occurred while executing the generated code: {traceback.format_exc()}")
        return None

def refine_response(user_query, generated_result):
    # Refine the raw data output into a more human-readable format.
    prompt = f'''
You are a chatbot that refines raw data outputs to be more human-readable and user-friendly.

User Question: "{user_query}"

Raw Response:
{generated_result}

Instructions:
- Refine the response into a clear, well-written answer.
- Format the response in a more readable way, such as a bulleted or numbered list.
- Add a brief introductory statement.
- Ensure the response is concise and directly answers the user's question.
- Show numerical output without any currency symbols. show currency only if that column is related to any money related or currency .

Please generate the appropriate response.
'''

    refined_response = generate_text(prompt)
    return refined_response

def handle_user_query(user_query, df):
    generated_code = generate_code(user_query, df)

    if "I apologize" in generated_code:
        return None, "Please ask the question related to the uploaded CSV file."

    return generated_code, None

def main():
    st.sidebar.markdown("""
        <h1 style="font-size: 26px; text-align: center;">DataScribe</h1>
        <h2 style="font-size: 14px; text-align: center;">Chat with CSV</h2>
    """, unsafe_allow_html=True)

    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Add an initial message
        st.session_state.messages.append({"role": "assistant", "content": "Please upload the CSV file to start analyzing your data."})

    if prompt := st.chat_input("Ask your question about the data..."):
        if not uploaded_file:
            st.error("Please upload the CSV file first.")
            st.session_state.messages.append({"role": "assistant", "content": "Please upload the CSV file first."})
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})

            df = extract_text_from_csv(uploaded_file)
            if df is not None:
                generated_code, warning_message = handle_user_query(prompt, df)

                if generated_code:
                    generated_result = execute_generated_code(generated_code, df)
                    refined_response = refine_response(prompt, generated_result)
                    st.session_state.messages.append({"role": "assistant", "content": refined_response})
                else:
                    st.session_state.messages.append({"role": "assistant", "content": warning_message})
            else:
                st.session_state.messages.append({"role": "assistant", "content": "Error processing the file."})

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if __name__ == '__main__':
    main()
