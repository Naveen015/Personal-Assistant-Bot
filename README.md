# Personal-Assistant-Bot

This project features a conversational AI designed to act as a personal assistant. It involves fine-tuning a language model on a custom dataset and then providing an interactive chat interface for user interaction.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Prerequisites](#prerequisites)
- [Setup Instructions](#setup-instructions)
  - [1. Clone the Repository](#1-clone-the-repository)
  - [2. Python Environment Setup](#2-python-environment-setup)
  - [3. Git Large File Storage (Git LFS) Setup](#3-git-large-file-storage-git-lfs-setup)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [1. Data Preparation](#1-data-preparation)
  - [2. Fine-tuning the Model](#2-fine-tuning-the-model)
  - [3. Running the Chat Interface](#3-running-the-chat-interface)
- [License](#license)

---

## Project Overview
This system leverages a fine-tuned AI model to provide conversational assistance. The project is structured into two main phases:
1.  **Fine-tuning:** Training a base language model on a specific dataset (`data.jsonl`) to equip it with personal assistant capabilities. This process generates large model checkpoint files.
2.  **Chat Interface:** An interactive Python script (`chat.py`) that allows users to communicate directly with the fine-tuned personal assistant bot.

---

## Prerequisites
Before you begin, ensure you have the following installed on your system:

* **Python 3.x:** (Python 3.8 or newer is recommended)
* **Git:** Version control system.
* **Git Large File Storage (Git LFS):** This is absolutely essential for handling the large model files generated during fine-tuning.

---

## Setup Instructions

### 1. Clone the Repository
If you haven't already, clone this repository to your local machine:

```bash
git clone [https://github.com/Naveen015/Personal-Assistant-Bot.git](https://github.com/Naveen015/Personal-Assistant-Bot.git)
cd Personal-Assistant-Bot
````

*Note: If you've been setting up an existing local project as a monorepo, you likely already have these files on your system.*

### 2\. Python Environment Setup

It's highly recommended to use a **Python virtual environment** to manage project dependencies in isolation and avoid conflicts with other Python projects.

a. **Create a virtual environment:**

```bash
python -m venv env
```

b. **Activate the virtual environment:**

  * **On Windows:**
    ```bash
    .\env\Scripts\activate
    ```
  * **On macOS/Linux:**
    ```bash
    source env/bin/activate
    ```

c. **Install required Python packages:**
After activating your environment, install the necessary libraries. This assumes you have a `requirements.txt` file in your project's root directory listing all dependencies.

```bash
pip install -r requirements.txt
```

*(If you don't have a `requirements.txt` file, you'll need to manually install dependencies like `transformers`, `torch`, `pandas`, etc., based on your project's specific requirements for the bot.)*

### 3\. Git Large File Storage (Git LFS) Setup

This project contains large files (such as fine-tuned model checkpoints, e.g., `.safetensors`, `.pt` files) which are managed by Git LFS. You **must** set up Git LFS to properly download and work with these files.

a. **Install Git LFS:**

  * **macOS (using Homebrew):**

    ```bash
    brew install git-lfs
    ```

  * **Debian/Ubuntu:**

    ```bash
    sudo apt-get install git-lfs
    ```

  * **Windows (using Chocolatey, if installed):**

    ```bash
    choco install git-lfs
    ```

    *(If `choco install git-lfs` encounters issues, you might need to install Git for Windows directly first, ensuring Git LFS is an included component, or download the Git LFS installer directly from [https://git-lfs.github.com/](https://git-lfs.github.com/) and run it.)*

  * **Other OS / Direct Download:** Visit the official Git LFS website for direct download and installation instructions: [https://git-lfs.github.com/](https://git-lfs.github.com/)

b. **Initialize Git LFS for this repository:**
Navigate to the root of your project directory (`Personal-Assistant-Bot`) in your terminal and run:

```bash
git lfs install
```

c. **Track large file types:**
Make sure Git LFS is tracking the necessary large files. This project explicitly uses `.pt` (PyTorch checkpoint) and `.safetensors` (Hugging Face SafeTensors) files.

```bash
git lfs track "*.pt"
git lfs track "*.safetensors"
```

This command creates or updates the `.gitattributes` file.

d. **Commit `.gitattributes` (if not already committed):**
It's crucial to commit the `.gitattributes` file so Git knows which files are managed by LFS across different collaborators.

```bash
git add .gitattributes
git commit -m "Add Git LFS tracking for model files"
```

e. **Pull LFS files (if needed):**
If you've just cloned the repository and notice pointer files instead of actual large files, or if you just set up LFS, ensure the actual large files are downloaded:

```bash
git lfs pull
```

-----

## Project Structure

  - `finetune_model.py`: The main script to train and fine-tune the AI model for personal assistant capabilities.
  - `chat.py`: An interactive script to run the chat interface using the fine-tuned model.
  - `data.jsonl`: The dataset in JSON Lines format used for fine-tuning the model.
  - `my_finetuned_chatbot_adapters/`: (This directory will be created by `finetune_model.py`) This is where the fine-tuned model's adapter weights and checkpoints (e.g., `adapter_model.safetensors`, `optimizer.pt`) will be saved.

-----

## Usage

### 1\. Data Preparation

Ensure your fine-tuning data is prepared and available in the `data.jsonl` file, located in the root directory of this project. This file should be in the [JSON Lines format](https://jsonlines.org/), with each line being a valid JSON object representing a training example for your personal assistant bot.

### 2\. Fine-tuning the Model

This step will train your AI model using the provided `data.jsonl` and save the fine-tuned checkpoints. This process can be resource-intensive and will create the large files that Git LFS manages.

**Before running this, make sure your Python environment is activated (refer to Step 2.b in Setup).**

```bash
python finetune_model.py
```

Once completed, the fine-tuned model's adapters and checkpoints will be saved within the `my_finetuned_chatbot_adapters/` directory.

### 3\. Running the Chat Interface

After the model has been successfully fine-tuned and its checkpoints are saved, you can run the interactive chat interface to communicate with your personal assistant bot.

**Ensure your Python environment is activated and the model has been fine-tuned (refer to Step 2.b in Setup and Step 2 in Usage).**

```bash
python chat.py
```

Follow the prompts in your terminal to interact with your AI personal assistant.

-----

## License

MIT License

Copyright (c) 2025 Naveen Prashanna Gurumurthy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.