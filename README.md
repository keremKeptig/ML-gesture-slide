
## Project Overview

This project implements a full pipeline for gesture recognition using pose and landmark data. It includes:

- A neural network trained from scratch
- PCA-based dimensionality reduction
- Hyperparameter tuning
- Model evaluation and visualization
- A slideshow presentation powered by Reveal.js
- A live demo for real-time gesture prediction via webcam



## Project Structure

The project structure is as follows:

- `data/`: All the training and validation data. Two separate folders for mandatory and optional gestures.

- `evaluation/` : Contains performance summaries and visualizations:
  `hyperparameter_results_mandatory.csv`: Results of hyperparameter tuning for mandatory gestures.
  - `hyperparameter_results_optional.csv`: Results of hyperparameter tuning for optional gestures.
  - `visualization.ipynb`: Notebook to analyze and visualize training/evaluation metrics 


- `presentation/`: Contains our presenation video and the powerpoint of the presentation

- `models/`: The best models for our mandatory and optional gestures and the corresponding PCA parameters (`best_model_mandatory.npz`, `best_model_optional.npz`, `pca_params.pkl`).

- `performance_score/`: here we can test our model in prediction mode ,so you need to supply the csv file

- `slideshow/`: This folder runs the slideshow via Reveal.js. See Slideshow Server below for instructions.

- `framework/`: All source code:
  - `model.py`: Implements a customizable feedforward neural network from scratch.
    - Activation Function: ReLU is used for all hidden layers.
    - Output Activation: Softmax is used for the final classification layer.
  - `train.py`: Handles training logic, batch updates, early stopping, PCA saving, and model persistence.
    - Loss Functions Supported:
      - `cross_entropy`: For multi-class classification (default).
      - `mse`: Mean Squared Error alternative option, though less ideal for classification.
  - `pca.py`: Implementation of PCA including explained variance plotting, saving/loading, and transform utilities.
  - `preprocess.py`: Loads, cleans, maps, and prepares gesture data for training, and applies PCA if configured.
  - `evaluation.py`: Includes accuracy, precision, recall, F1-score, confusion matrix computation, and visualization.
  - `hyperparameter_search.py`: Performs grid search for neural network hyperparameters, logs and visualizes results.

- `demo.ipynb` and `demo_optional.ipynb`: The notebooks used to train the current version of our models. 

### Prediction Mode
you can use the prediction mode with:

 `python performance_score/log-events-script.py --input_csv=INPUT_CSV --output_csv_name=RESULTS_CSV`     
 
 And then the same input file for ground_truth and for the events created output file:

`python performance_score/calculator.py --events_csv=RESULTS_CSV --ground_truth_csv=INPUT_CSV`

After that we can visualize these results with following command:

`python performance_score/events_visualization.py --events_csv=RESULTS_CSV --ground_truth_csv=INPUT_CSV`

## How to use the slideshow folder
First, you need to make sure that you install all the requirements with `pip install -r requirements.txt`


This project includes a slideshow powered by Reveal.js and a real-time gesture prediction demo.


### Run the Slideshow Server

The slideshow is built using HTML/JS (Reveal.js), so you'll need to run a local HTTP server:

`python slideshow_demo.py`

Then you need to go to the http://localhost:8000

Once the slideshow is running, launch the live demo script that performs gesture recognition using the trained model:

`python live_mode.py`


![Architecture:](images/diagrampng)









