# Implicit Hate Speech Data Generator

A modular pipeline to generate implicit hate speech variations, benign counterfactuals, and their corresponding implied meanings using Large Language Models (LLMs).

## 📂 Project Structure
* **`main.py`**: The entry point. Runs the generation loop.
* **`config.py`**: Settings (Model IDs, Seed Data, Hardware Toggles).
* **`prompts.py`**: LangChain templates for generation and logic extraction.
* **`utils.py`**: GPU management and model loading logic.

## 🚀 How to Run

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/aamitssharma07/CoRe-Hate.git
    ```

2.  **Set Up Virtual Environment**
    ```bash
    # Create the virtual environment
    python3 -m venv .venv
    
    # Activate it
    source .venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Pipeline**
    ```bash
    python main.py
    ```
    *Tip: If running on a remote server, use `nohup python main.py &` to keep it running in the background.*

## ⚙️ Configuration
Open **`config.py`** to change:
* `MODELS_TO_TEST`: Add or remove HuggingFace model IDs (e.g., Mistral, Vicuna).
* `SEED_POSTS`: Add new seed hate speech examples.


## 📊 Output
Results are saved automatically in the **`Results/`** folder as CSV files (e.g., `results_mistralai_Mistral-7B.csv`).

**Data Columns:**
* **`id`**: Unique identifier (e.g., `1` = Seed, `1.1` = Variation, `1.1_cf` = Counterfactual).
* **`text`**: The generated post content.
* **`implied_statement`**: The specific logic or meaning behind that post.
* **`category`**: Type of row (`seed_original`, `generated_hate`, `counterfactual`).