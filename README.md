# NLP_Project
# Bias Detector for English Text

A Natural Language Processing (NLP) project that automatically detects bias in English text using a fine-tuned **RoBERTa** model with **LoRA (Low-Rank Adaptation)**. The system classifies text into one of four categories: **Gender Bias, Racial Bias, Political Bias,** or **No Bias**.

## Project Overview

Bias in online content can influence opinions, reinforce stereotypes, and spread misinformation. This project aims to automatically identify different types of bias in English text such as news articles, tweets, and paragraphs using a transformer-based deep learning model.

## Features

* Detects multiple types of bias in English text
* Fine-tuned **RoBERTa-base** model for sequence classification
* Uses **LoRA (Low-Rank Adaptation)** for parameter-efficient fine-tuning
* Four-class classification:

  * Gender Bias
  * Racial Bias
  * Political Bias
  * No Bias

## Example

**Input**

> Women are naturally bad at coding.

**Output**

```
Predicted Bias: Gender Bias
Confidence:
Gender Bias   : 98.53%
Political Bias: 1.43%
No Bias       : 0.03%
Racial Bias   : 0.01%
```

## Dataset

This project uses the **Media Bias (MBIB)** dataset from Hugging Face.

The dataset contains English text collected from news articles, tweets, and other sources and is divided into:

* Training Set: 72%
* Validation Set: 8%
* Test Set: 20%

## Model Architecture

* Pretrained **RoBERTa-base**
* Byte Pair Encoding (BPE) Tokenizer
* LoRA adapters applied to the attention Query and Value matrices
* Sequence Classification Head
* Cross-Entropy Loss

## Technologies Used

* Python
* Google Colab
* Hugging Face Transformers
* PEFT (LoRA)
* PyTorch
* NumPy
* Pandas
* Matplotlib

## Performance

Evaluation Results:

| Metric    |      Score |
| --------- | ---------: |
| Accuracy  | **76.36%** |
| Precision | **73.71%** |
| Recall    | **76.36%** |
| F1 Score  | **71.24%** |

## Repository Structure

```
.
├── README.md
├── nlp.ipynb
├── config.json
├── tokenizer.json
├── tokenizer_config.json
├── adapter_config.json
├── adapter_model.safetensors
├── model.safetensors
└── LICENSE
```

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Bias-Detector-for-English-Text.git
cd Bias-Detector-for-English-Text
```

Install the required packages:

```bash
pip install transformers peft datasets torch numpy pandas matplotlib
```

## Usage

1. Open `nlp.ipynb` in Google Colab or Jupyter Notebook.
2. Install the required dependencies.
3. Load the pretrained model and tokenizer.
4. Run the notebook to train or predict bias in English text.

## Future Improvements

* Improve classification accuracy with additional fine-tuning.
* Support multilingual bias detection.
* Deploy the model as a web application using Streamlit or Flask.
* Add explainability techniques such as attention visualization.

## References

* RoBERTa-base
* Hugging Face Transformers
* PEFT (LoRA)
* Media Bias (MBIB) Dataset
* *Attention Is All You Need* (Vaswani et al., 2017)

## Author

**M. Roja Lakshmi**

B.Tech in Computer Science and Engineering

National Institute of Technology Andhra Pradesh
