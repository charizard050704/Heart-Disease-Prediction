# Deployment Guide

## Local Deployment
1. Clone or extract the project folder.
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Mac/Linux
   .venv\Scripts\activate    # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Train the model:
   ```bash
   python scripts/train.py
   ```
5. Run the Streamlit app:
   ```bash
   streamlit run app/heart_ui_app.py
   ```

## Streamlit Cloud Deployment
1. Push project to GitHub.
2. Go to [streamlit.io](https://streamlit.io/cloud).
3. Deploy new app from your GitHub repo.
4. Set working directory and Python version.
5. App will be available with a public link.
