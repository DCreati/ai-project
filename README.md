# Dynamic-RAG

## Risorse

- 2WikiMultihopQA: https://www.aclweb.org/anthology/2020.coling-main.580/
- DRAGIN: https://arxiv.org/abs/2403.10081
- RAG Survey: https://arxiv.org/abs/2402.19473


## Overview  DRAGIN: Dynamic Retrieval Augmented Generation

**DRAGIN** (Dynamic Retrieval Augmented Generation based on the real-time Information Needs of LLMs) is a framework designed to enhance the text generation capabilities of Large Language Models (LLMs) by dynamically deciding when and what to retrieve during the generation process. This approach addresses the limitations of existing dynamic RAG methods, improving the quality and relevance of generated text.

## Table of Contents

- [Core Components](#core-components)
  - [dragin/ Directory](#dragin-directory)
    - [__init__.py](#initpy)
    - [config.py](#configpy)
    - [model.py](#modelpy)
    - [retriever.py](#retrieverpy)
    - [utils.py](#utilspy)
  - [data/ Directory](#data-directory)
    - [prep_elastic.py](#prep_elasticpy)
  - [train.py](#trainpy)
  - [evaluate.py](#evaluatepy)
  - [requirements.txt](#requirementstxt)
  - [README.md](#readmemd)
  - [tests/ Directory](#tests-directory)
- [Installation Instructions](#installation-instructions)
- [Usage](#usage)
- [Contributing](#contributing)

## Core Components

### dragin/ Directory

This directory contains the main implementation of the DRAGIN framework.

#### __init__.py

- **Goal:** Marks the directory as a Python package and can be used to initialize package-level variables or imports.
- **Structure:** Typically empty or contains import statements for modules within the package.

#### config.py

- **Goal:** Contains configuration settings for the DRAGIN framework, including model parameters, dataset paths, and other hyperparameters.
- **Structure:**
  - Defines a configuration class or functions to load and parse configuration files (e.g., JSON or YAML).
  - Example parameters include model name, dataset paths, and training hyperparameters.

#### model.py

- **Goal:** Implements the DRAGIN model architecture, integrating the LLM and the retrieval mechanism.
- **Structure:**
  - **Class Definition:** Defines the main model class, which includes methods for forward propagation, retrieval activation, and query formulation.
  - **Methods:**
    - `forward()`: Handles input data and generates output using the LLM.
    - `retrieve()`: Manages the retrieval process based on the model's current state and input.

#### retriever.py

- **Goal:** Implements the retrieval mechanism that interacts with external data sources (e.g., Elasticsearch).
- **Structure:**
  - **Class Definition:** Defines a retriever class that handles query formulation and data retrieval.
  - **Methods:**
    - `query()`: Takes a formulated query and retrieves relevant documents from the index.
    - `index()`: Indexes new documents into the retrieval system.

#### utils.py

- **Goal:** Contains utility functions that support various operations within the framework, such as data preprocessing and evaluation metrics.
- **Structure:**
  - **Functions:**
    - `load_data()`: Loads datasets from specified paths.
    - `evaluate()`: Computes evaluation metrics for model performance.

### data/ Directory

This directory contains scripts and utilities for handling datasets used in training and evaluation.

#### prep_elastic.py

- **Goal:** Prepares and indexes the Wikipedia dataset into Elasticsearch for retrieval.
- **Structure:**
  - **Main Function:** Handles downloading, preprocessing, and indexing of the dataset.
  - **Command-line Interface:** Allows users to specify parameters like data path and index name.

### train.py

- **Goal:** The main script for training the DRAGIN model.
- **Structure:**
  - **Argument Parsing:** Uses libraries like `argparse` to handle command-line arguments for training configurations.
  - **Training Loop:** Implements the training process, including data loading, model training, and logging.
  - **Checkpointing:** Saves model checkpoints at specified intervals.

### evaluate.py

- **Goal:** Evaluates the performance of the trained DRAGIN model on specified datasets.
- **Structure:**
  - **Argument Parsing:** Similar to `train.py`, it handles command-line arguments for evaluation settings.
  - **Evaluation Loop:** Loads the model and dataset, performs inference, and computes evaluation metrics.

### requirements.txt

- **Goal:** Lists the Python packages required to run the DRAGIN framework.
- **Structure:** Contains package names and versions, ensuring that users can replicate the environment.

### README.md

- **Goal:** Provides an overview of the DRAGIN project, including installation instructions, usage examples, and a brief description of the framework.
- **Structure:**
  - **Sections:** Introduction, Installation, Usage, and
