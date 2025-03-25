# Sentiment_Analysis

# Sentiment Analysis Using BERT

This project implements a sentiment analysis model using BERT (Bidirectional Encoder Representations from Transformers) to classify IMDB movie reviews as either positive or negative. The model is trained using a custom dataset and saved for future inference.

## Project Structure
- **`Dataset/`**: Contains the IMDB movie reviews dataset.
- **`Model_Training.ipynb`**: Jupyter notebook used for training the BERT model. After training, the model is saved in the `Model/` folder.
- **`Model/`**: Directory where the trained model is stored (`bert_sentiment_model.pt`).
- **`Sentiment_Analysis.py`**: Python script to load the saved model and perform sentiment analysis on user-provided reviews.

## Getting Started

### Prerequisites
Before running the project, ensure that you have the following packages installed:
- `transformers`
- `torch`
- `pandas`
- `scikit-learn`
- `numpy`
- `flask` (if you plan to deploy it)
- `jupyter` (for running the `.ipynb` notebook)

You can install all the dependencies using:
bash
pip install transformers torch pandas scikit-learn numpy flask jupyter

### How to Run

1. **Dataset Preparation**: Place the IMDB dataset in the `Dataset/` folder.
2. **Model Training**:
   - Open `Model_Training.ipynb` in Jupyter Notebook.
   - Run the cells sequentially to preprocess the data, train the BERT model, and save it to the `Model/` folder as `bert_sentiment_model.pt`.
3. **Sentiment Analysis**:
   - Once the model is saved, you can run `Sentiment_Analysis.py` from the command line:
   ```bash
   python Sentiment_Analysis.py
   - This script will load the saved model and prompt you to enter a movie review. It will then output the sentiment prediction (positive or negative).



## Project Workflow

1. **Data Preprocessing**: Text cleaning, removal of unwanted characters, and limiting review length.
2. **Model Training**: Fine-tuning the BERT model for sentiment classification using the IMDB reviews.
3. **Saving the Model**: After training, the model is saved in the `Model` folder.
4. **Sentiment Prediction**: Using the trained model to classify new reviews as positive or negative.

## Future Plans
The current version of the project is designed for command-line usage. Future updates may include:
1. **Deployment**: Using `Flask` or `FastAPI` for web-based deployment.
2. **Enhanced Preprocessing**: Further refinement of text preprocessing to handle sarcasm or contextual sentiments better.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
```
