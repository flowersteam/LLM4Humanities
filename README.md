# LLM4Humanities

**LLM4Humanities** is a **Python-based toolkit** and [**web app**](https://flowanalysis.streamlit.app/) for assisting qualitative researchers in annotating textual data using **Large Language Models (LLMs)**.

It provides an end-to-end workflow that combines **manual coding**, **automated classification**, and **evaluation metrics** to use and help you decide whether LLMs can reliably support your annotation tasks.

## Key Features

- No coding skills required using the app
- Supports **manual** and **automatic** annotation of qualitative data
- Built-in evaluation metrics to compare human and model agreement
- Designed for **multi-class**, and **binary** classification
- Compatible with **OpenRouter**, **OpenAI**, **Anthropic**, **Gemini** and **Azure** API keys

# Table of Contents
- [Why this tool](#why-this-tool)
- [How it works](#how-it-works)
- [Data requirements](#data-requirements)
- [Running without installation](#running-without-installation)
- [Run Locally (Full Control)](#run-locally-full-control)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Cite / References](#cite--references)

## Why this tool

Most datasets require at least one human coder to manually classify data before any analysis can begin. This process relies on a carefully defined set of guidelines (a codebook) to assign data to categories.
**Annotation** is therefore indispensable in experimental psychology and related fields, but it is also slow and costly. Moreover, best practice requires at least two independent annotators to compute **inter-rater reliability** (IRR), making the process even more resource-intensive.

IRR is crucial in experimental psychology because it separates the variability of human judgments from the signal researchers aim to interpret. Without IRR, it is impossible to assess what portion of the data variance reflects coder idiosyncrasies rather than the phenomenon under study, putting the validity of scientific conclusions at risk.
Yet, despite its importance, IRR is often skipped, misreported, or calculated with inadequate metrics (e.g., raw % agreement, Pearson correlation).

**LLM4Humanities** addresses these challenges with a complete pipeline:  

1. **Manually annotate** only a small portion of your data *(minimizing cost and effort).*  
2. **Use an LLM** to classify the rest of your dataset.  
3. **Compare** LLM predictions to the human-labeled subset using proper IRR metrics, confidence intervals, and decision guidance.  

## How It Works

### 1. **Manual Annotation**
Use the [**Manual annotator App**](https://datannotate.streamlit.app/) to label a subset of your dataset if you need to.  
This serves as a reference to evaluate model performance.

### 2. **Automated Annotation**

You can choose between:

- **Web App** [**(LLM4Humanities)**](https://flowanalysis.streamlit.app/):  
  - Configure prompts and LLM providers  
  - Run classification scenarios on your labeled subset  
  - Measure agreement between model and human coders  

- **Step-by-Step Notebooks** (Google Colab):  
  - Guides you through the same workflow in code cells  
  - Requires minimal coding knowledge—each step is clearly explained  
  - Ideal if you prefer more control, customization, or wish to inspect intermediate results 
  - Examples dataset are provided 

### 3. **Evaluate Model Performance**
We provide several metrics for you to use as you wish to make your own informed choice (in the app or in the notebooks):

- **Cohen's Kappa**  
  Measures agreement between the LLM and human annotators.

- **Krippendorff's Alpha**  
  For ordinal or nominal labels, compute confidence intervals via bootstrapping.

- **ALT-Test** ([arXiv:2501.10970](https://arxiv.org/pdf/2501.10970))  
  A robust non-inferiority test comparing the model to each annotator in turn.  
  Requires **at least 3 human annotators**.

- **Classification Metrics**  
  Per-class breakdown of true/false positives, recall, and error rates.

## Data Requirements

Your dataset should be in **CSV or Excel** format, with:

- One row per entry to classify
- One or more **textual columns** that will be shown to the model
- At least one **unique identifier** column
- For evaluate the LLM classification, at least one **annotation column** for human labels

## Running without installation

### **Use the web interface**

Click the link below to run the **manual annotator**: 

[Run the Manual annotator App](https://datannotate.streamlit.app/)

Click the link below to run the **web app**: 

[Run the LLM4Humanities App](https://flowanalysis.streamlit.app/)

### **Run in Google Colab**

Click the badge below to run the notebooks directly in **Google Colab**:

Binary classification notebook:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OlivierLClerc/qualitative_analysis_project/blob/master/notebooks/binary_case_colab.ipynb)

Multiclass classification notebook:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OlivierLClerc/qualitative_analysis_project/blob/master/notebooks/multiclass_case_colab.ipynb)

Complex classification notebook:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OlivierLClerc/qualitative_analysis_project/blob/master/notebooks/complex_case_colab.ipynb)

## **Run Locally (Full Control)**

If you prefer to run the analysis directly on your machine, follow these installation steps.

1. Clone the repository.

```bash
git clone https://github.com/OlivierLClerc/qualitative_analysis_project.git
cd qualitative_analysis_project
```
2. Create a Virtual Environment

```bash
conda create -n qualitative_analysis python=3.10
conda activate qualitative_analysis
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```
4. Set up your API credentials.

Copy or rename `.env.example` to `.env`. Populate it with your LLM credentials (OpenAI, Azure, or Together keys and endpoints).

Example:

```bash
# OpenAI
OPENAI_API_KEY=your_openai_key_here

# Azure environment variables
AZURE_API_KEY=your_azure_api_key_here
AZURE_OPENAI_ENDPOINT=your_azure_endpoint_here
AZURE_API_VERSION=your_azure_api_version_here

# Together AI Configuration
TOGETHERAI_API_KEY=your_togetherai_api_key_here
```

5. Run the apps or notebooks

## Usage

To run the Manual Annotator, use:

```bash
streamlit run manual_annotator_stream.py
```

To run the LLM4Humanities app, use:

```bash
streamlit run app.py
```

## Project Structure

```
qualitative_analysis_project/
├── data/
│   ├── binary_user_case/
│   │   ├── outputs/
│   │   │   ├── binary_case_for_metrics.csv
│   │   ├── binary_config.json
│   │   └── binary_data.xlsx
│   ├── complex_user_case/
│   │   ├── outputs/
│   │   │   ├── complex_case_for_metrics.csv
│   │   ├── complex_config.json
│   │   └── complex_data.xlsx
│   ├── multiclass_user_case/
│   │   ├── outputs/
│   │   │   ├── multiclass_case_for_metrics.csv
│   │   ├── multiclass_config.json
│   │   └── multiclass_data.xlsx
│   └── outputs/
├── manual_annotator/
│   ├── __init__.py
│   ├── annotation_filter.py
│   ├── annotator_setup.py
│   ├── app_core.py
│   ├── codebook_upload.py
│   ├── column_selection.py
│   ├── data_download.py
│   ├── data_upload.py
│   ├── label_definition.py
│   └── row_annotation.py
├── notebooks/
│   ├── binary_case_colab.ipynb
│   ├── complex_case_colab.ipynb
│   └── multiclass_case_colab.ipynb
├── qualitative_analysis/
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── alt_test.py
│   │   ├── classification.py
│   │   ├── kappa.py
│   │   ├── krippendorff.py
│   │   ├── utils.py
│   │   └── visualization.py
│   ├── __init__.py
│   ├── config.py
│   ├── cost_estimation.py
│   ├── data_processing.py
│   ├── logging.py
│   ├── model_interaction.py
│   ├── notebooks_functions.py
│   ├── parsing.py
│   ├── prompt_engineering.py
│   └── scenario_runner.py
├── streamlit_app/
│   ├── __init__.py
│   ├── analysis.py
│   ├── app_core.py
│   ├── codebook_management.py
│   ├── column_selection.py
│   ├── data_upload.py
│   ├── evaluation.py
│   ├── field_selection.py
│   ├── llm_configuration.py
│   ├── prompt_construction.py
│   └── session_management.py
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml
├── app.py
├── manual_annotator_stream.py
├── README.md
└── requirements.txt
```

### Main Files and Folders

#### `app.py`
The main entry point for the LLM4Humanities app. It imports the modularized app from the streamlit_app package.

#### `manual_annotator_stream.py`
The main entry point for the Manual Annotator app. It imports the ManualAnnotatorApp class and sets up the Streamlit interface.

#### `manual_annotator/`
Contains the modules for the Manual Annotator app.

#### `streamlit_app/`
Contains the modules for the LLM4Humanities app.

#### `notebooks/`
Contains Jupyter notebooks demonstrating user-case for:
- Binary classification
- Multiclass classification
- Complex classification

#### `data/`
Holds sample data files for the different classification scenarios, organized into user case directories, plus an `outputs/` subfolder where processed results can be saved.

#### `qualitative_analysis/`
The main Python package containing modules used by the apps and the notebooks for:
- Configuration management
- Data processing
- Model interaction
- Prompt construction and engineering
- Evaluation metrics
- Cost estimation
- Logging

#### `codebook/`
Contains text files defining classification rules or codebooks for the user-case.

#### Other Files

- **`.env.example`** – Template for environment variables needed for API credentials.
- **`.pre-commit-config.yaml`** – Configuration for pre-commit hooks (linting, formatting, etc.).
- **`.gitignore`** – Specifies files to be ignored by Git.
- **`requirements.txt`** – Lists all Python dependencies required for the project.

## License

This project is licensed under the MIT License.
