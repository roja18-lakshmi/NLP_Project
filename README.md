# Bias Detector for English Text using RoBERTa and LoRA

A Natural Language Processing (NLP) project that detects bias in English text using a fine-tuned **RoBERTa** model with **LoRA (Low-Rank Adaptation)**. The model classifies text into one of four categories: **Gender Bias**, **Racial Bias**, **Political Bias**, or **No Bias**.

---

## Overview

Bias in online content can influence opinions, reinforce stereotypes, and spread misinformation. This project presents an automated bias detection system that identifies different types of bias in English text such as news articles, tweets, and paragraphs.

The project uses the **RoBERTa-base** transformer model along with **LoRA (Low-Rank Adaptation)** for efficient fine-tuning, significantly reducing the number of trainable parameters while maintaining strong performance.

---

## Features

* Detects bias in English text
* Four-class classification:

  * Gender Bias
  * Racial Bias
  * Political Bias
  * No Bias
* RoBERTa-base transformer model
* LoRA-based parameter-efficient fine-tuning
* Developed using Google Colab
* Comparison between LoRA fine-tuning and training only the classification head

---

## Dataset

The project uses the **Media Bias (MBIB)** dataset from Hugging Face.

The dataset contains English text collected from:

* News articles
* Tweets
* Other written content

Dataset split:

* **Training Set:** 72%
* **Validation Set:** 8%
* **Test Set:** 20%

---

## Model Architecture

The model consists of the following stages:

1. **Tokenization**

   * RoBERTa tokenizer
   * Byte Pair Encoding (BPE)
   * Padding and truncation to a sequence length of 128

2. **Transformer Encoder**

   * Pretrained RoBERTa-base model
   * Produces contextual embeddings of hidden size 768

3. **LoRA Fine-Tuning**

   * Applied to the Query and Value attention matrices
   * Remaining RoBERTa parameters remain frozen
   * Classification head remains trainable

4. **Classification Layer**

   * Fully connected layer
   * Softmax activation for four-class prediction

5. **Loss Function**

   * Cross-Entropy Loss

---

## Performance Comparison

### 1. RoBERTa + LoRA

| Metric    |      Value |
| --------- | ---------: |
| Accuracy  | **76.09%** |
| Precision | **72.81%** |
| Recall    | **76.09%** |
| F1 Score  | **70.43%** |

Trainable Parameters:

* **888,580**
* Only **0.7078%** of the total model parameters were trained.

---

### 2. Training Only the Classification Head

| Metric    |      Value |
| --------- | ---------: |
| Accuracy  | **71.74%** |
| Precision | **60.39%** |
| Recall    | **71.74%** |
| F1 Score  | **64.83%** |

Trainable Parameters:

* **593,668**
* Only **0.4763%** of the total model parameters were trained.

---

## Results

The experimental results demonstrate that applying **LoRA** to the Query and Value attention matrices improves overall classification performance compared to training only the classification head.

The LoRA-based model achieved higher Accuracy, Precision, Recall, and F1-score while requiring only a small percentage of the model parameters to be updated during fine-tuning.

---

## Example Prediction

### Input

```text
Women are naturally bad at coding.
```

### Output

```text
Predicted Bias : Gender Bias

Probabilities

No Bias        : 0.00008
Gender Bias    : 0.98535
Political Bias : 0.01429
Racial Bias    : 0.00027
```

---

## Technologies Used

* Python
* Google Colab
* Hugging Face Transformers
* PEFT (LoRA)
* PyTorch
* NumPy
* Pandas
* Matplotlib

---

## Repository Structure

```text
Bias-Detector-for-English-Text/
│
├── README.md
├── nlp.ipynb
├── config.json
├── tokenizer.json
├── tokenizer_config.json
├── adapter_config.json
├── adapter_model.safetensors
├── model.safetensors
├── LICENSE
└── images/
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Bias-Detector-for-English-Text.git
```

Move into the project directory:

```bash
cd Bias-Detector-for-English-Text
```

Install the required libraries:

```bash
pip install transformers peft datasets torch numpy pandas matplotlib
```

---
## Model Weights

The complete model weights (`model.safetensors`) are not included in this repository because the file exceeds GitHub's file size limit.

To use the model:

- Train the model using `nlp.ipynb`, or
- Download the model weights from the provided link (if available).

## Usage

1. Open **nlp.ipynb** using Google Colab or Jupyter Notebook.
2. Install the required dependencies.
3. Load the pretrained tokenizer and model.
4. Run the notebook to train the model or predict bias in English text.

---

## Future Work

* Improve classification accuracy using larger datasets.
* Support multilingual bias detection.
* Deploy the model as a web application.
* Provide explainable predictions using attention visualization techniques.

---

## References

* RoBERTa-base
* Hugging Face Transformers
* PEFT (LoRA)
* Media Bias (MBIB) Dataset
* *Attention Is All You Need* – Vaswani et al., 2017
* *Speech and Language Processing* – Daniel Jurafsky

---

## Author

**M. Roja Lakshmi**

B.Tech in Computer Science and Engineering

National Institute of Technology Andhra Pradesh

---

⭐ If you found this project useful, consider giving it a star.
