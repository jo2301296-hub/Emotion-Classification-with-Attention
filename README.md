# Emotion Classification with Attention

A Natural Language Processing (NLP) project that classifies the emotion
expressed in a text into one of six categories. The project explores
traditional recurrent neural network architectures, an attention-based
model, and a fine-tuned Transformer model.

## Project Overview

The system is designed to recognize the following emotions:

-   **Joy**
-   **Sadness**
-   **Anger**
-   **Fear**
-   **Surprise**
-   **Disgust**

The project compares multiple deep learning approaches to understand how
sequence models, attention mechanisms, and pre-trained language models
perform on emotion classification.

## Dataset

This project uses the **GoEmotions** dataset released by Google
Research.

-   Dataset:
    [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions)
-   Kaggle source used for the project: [GoEmotions
    Dataset](https://www.kaggle.com/datasets/debarshichanda/goemotions)

GoEmotions contains fine-grained emotion annotations. For this project,
the data is prepared for a six-class classification task using the
target emotion groups listed above. The original train, validation
(development), and test splits are preserved.

> **Note:** GoEmotions is originally a multi-label dataset. The
> six-class experiment uses a filtered/mapped subset suitable for
> single-label classification. Results therefore apply to this prepared
> subset, not to the complete original multi-label task.

## Models Implemented

The project explores the following architectures:

1.  **LSTM** --- captures sequential dependencies in text.
2.  **GRU** --- a gated recurrent architecture with a simpler structure
    than LSTM.
3.  **BiLSTM with Attention** --- processes text in both directions and
    uses attention weights to emphasize relevant token representations.
4.  **DistilBERT** --- a pre-trained Transformer fine-tuned for the six
    emotion classes.

For the custom recurrent models, the project uses word embeddings
(including GloVe where configured). DistilBERT uses its contextual
Transformer representations and self-attention.

## DistilBERT Training Configuration

The DistilBERT experiment uses the Hugging Face checkpoint
`distilbert-base-uncased` with a custom PyTorch training loop.

  Parameter                                      Value
  ------------------------- --------------------------
  Maximum sequence length                          128
  Batch size                                        16
  Epochs                                             3
  Learning rate                                   2e-5
  Optimizer                                      AdamW
  Weight decay                                    0.01
  Model selection             Best validation Macro F1

The best model state is selected using validation Macro F1 and then
saved with its tokenizer.

## Evaluation

The models are evaluated using:

-   Accuracy
-   Macro F1-score
-   Weighted F1-score
-   Per-class Precision and Recall
-   Confusion Matrix

For the attention-based model, attention visualizations can also help
inspect which tokens contribute most strongly to a prediction.

**Results:** Add the final test-set metrics and model comparison table
here after running the complete evaluation. No performance values are
included in this README until they are confirmed from the actual test
results.

## Project Workflow

1.  Load the GoEmotions train, validation, and test splits.
2.  Map and filter the original emotion annotations for the six target
    classes.
3.  Clean and prepare the text and labels.
4.  Tokenize text or convert it to embeddings, depending on the model.
5.  Train LSTM, GRU, BiLSTM with Attention, and DistilBERT.
6.  Select models using validation performance.
7.  Evaluate the final models on the held-out test set.
8.  Compare metrics and inspect predictions.

## Technologies

-   Python
-   PyTorch
-   Hugging Face Transformers
-   Scikit-learn
-   NumPy
-   Pandas
-   Matplotlib / Seaborn
-   Jupyter Notebook / Kaggle

## Running the Project

The experiments were developed for a Kaggle Notebook environment.

1.  Open the project notebook in Kaggle.
2.  Add the GoEmotions dataset to the notebook.
3.  Make sure the dataset paths in the notebook match the Kaggle input
    directory.
4.  Run the notebook cells in order, from data preparation through
    evaluation.
5.  For GPU training, enable a GPU accelerator in the notebook settings
    when available.

Install the main dependencies if they are not already available:

``` bash
pip install torch transformers scikit-learn numpy pandas tqdm matplotlib seaborn
```

## Loading the Saved DistilBERT Model

The training notebook saves the selected DistilBERT model and tokenizer
to:

``` text
/kaggle/working/distilbert_emotion
```

They can be loaded with:

``` python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = "/kaggle/working/distilbert_emotion"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
```

The saved directory is local to the Kaggle session. Download the output
or save it as a Kaggle dataset/model artifact if it needs to be reused
in another session.

## Repository Structure

The exact repository structure may be updated as notebooks and
supporting files are added. A suggested organization is:

``` text
Emotion-Classification-with-Attention/
├── README.md
├── notebooks/
│   └── emotion_classification.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── models.py
│   ├── train.py
│   └── evaluate.py
└── results/
    ├── figures/
    └── model_comparison.csv
```

## Future Improvements

-   Add a user-friendly Gradio or Streamlit interface.
-   Display prediction confidence for each emotion.
-   Add attention heatmaps for more interpretable predictions.
-   Expand evaluation and error analysis across emotion categories.
-   Experiment with additional pre-trained language models.

## Acknowledgements

-   Google Research for the GoEmotions dataset.
-   Hugging Face for the Transformers library.
-   The open-source Python and PyTorch communities.

------------------------------------------------------------------------

**Project:** Emotion Classification with Attention\
**Focus:** Natural Language Processing, Deep Learning, Attention
Mechanisms, and Transformers# Emotion-Classification-with-Attention
