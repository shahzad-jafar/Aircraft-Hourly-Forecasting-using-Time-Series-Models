<<<<<<< 
Aircraft Hourly Forecasting using Time Series Models
This project provides a complete and modular pipeline for forecasting hourly Aircraft demand using various time-series models. This codebase has been refactored from Jupyter Notebooks into a structured Python project to make experimentation and deployment easier.

🚀 Setup and Installation
Follow the steps below to set up the project on your local machine.

Step 1: Create a Virtual Environment
First, open your terminal, navigate to the project folder, and create a virtual environment. This keeps your project's libraries separate from others.

python -m venv venv

Activate the environment:

On Windows: venv\Scripts\activate

On macOS/Linux: source venv/bin/activate

Step 2: Install Core Libraries
Now, install all the required libraries from the requirements.txt file.

pip install -r requirements.txt

Step 3: Install PyTorch Geometric Libraries
Some libraries (like torch-scatter) need to be installed separately according to your PyTorch version. Run the following commands in this order.

# First, check the PyTorch version and set it as an environment variable
export TORCH=$(python -c "import torch; print(torch.__version__)")

# Now, install the remaining libraries
pip install -q torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}.html
pip install -q torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}.html
pip install -q git+https://github.com/pyg-team/pytorch_geometric.git
pip install -q git+https://github.com/TorchSpatiotemporal/tsl.git

⚙️ How to Run the Project
After the setup is complete, you can run the project using the following steps.

Step 1: Data Preprocessing (One-Time Step)
First, you need to download and process the data. This command will create the data and preprocessed_data folders.

python preprocess.py

Step 2: Train a Model
Once the data is ready, you can train any model using the main.py script. Use the --model argument to specify the model's name.

To Train GraphWaveNet:
python main.py --model graphwavenet

To Train DCRNN:
python main.py --model dcrnn

To Train AutoGluon:
python main.py --model autogluon

🔧 Configuration
All model hyperparameters, directory paths, and training settings can be easily modified in the config.py file.

# Aircraft-Hourly-Forecasting-using-Time-Series-Models
This project focuses on forecasting aircraft data at an hourly level using advanced time series techniques. The pipeline includes data preprocessing, model training, and evaluation, enabling accurate prediction and analysis of aircraft behavior over time.

