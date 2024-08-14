# DataScribe: Chat with CSV

DataScribe is an interactive Streamlit application that allows you to chat with your CSV data. Simply upload a CSV file, ask questions, and get instant insights and code snippets generated using Google Gemini LLM.

## Features

- **CSV Upload**: Upload any CSV file for analysis.
- **Intelligent Query Handling**: Ask questions about your data, and the app will generate the appropriate Python code to answer them.
- **Code Execution**: Executes the generated code and returns results in a user-friendly format.
- **Refined Responses**: Get clear, concise, and formatted responses based on the raw output.

## Installation

To run this project locally, follow these steps:

### Prerequisites

- Python 3.8 or higher
- Git

### Clone the Repository

```bash
git clone https://github.com/your-username/DataScribe.git
cd DataScribe
```

### Set Up the Environment

1. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Set up Google API key**:

   Create a `.env` file in the root directory and add your Google API key:

   ```plaintext
   GOOGLE_API_KEY=your-google-api-key
   ```

   Alternatively, you can set the environment variable directly in your shell:

   ```bash
   export GOOGLE_API_KEY=your-google-api-key
   ```

### Running the App

```bash
streamlit run app.py
```

Open your browser and go to `http://localhost:8501` to view the application.

## Usage

1. Upload a CSV file using the file uploader in the sidebar.
2. Type a query related to the uploaded data in the chat input box.
3. View the results and generated code in the chat interface.

## Example Queries

- "What is the average value of the column `Sales`?"
- "Filter the rows where `Country` is `USA` and `Sales` is greater than 1000."
- "Show me a bar chart of `Product` against `Sales`."

## Project Structure

```plaintext
.
├── app.py                 # Main application script
├── requirements.txt       # List of required Python packages
├── README.md              # This file
├── .env.example           # Example environment variables file
└── .gitignore             # Git ignore file
```

## Contributing

If you'd like to contribute to this project, please fork the repository and use a feature branch. Pull requests are welcome.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Streamlit](https://streamlit.io/)
- [Pandas](https://pandas.pydata.org/)
- [Google Generative AI](https://cloud.google.com/generative-ai)
```

This version has proper formatting for sections, code blocks, and lists.
