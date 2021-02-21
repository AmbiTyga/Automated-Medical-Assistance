# Automated-Medical-Assistance:
## This repo contains the code for our paper: Automated Medical Assistance: Attention Based Consultation System (https://openreview.net/forum?id=jYV4ZXy0L5 "PDF")
## Authors
- [Raj Ratn Pranesh](https://rajratnpranesh.github.io/) & [Sumit Kumar](https://github.com/sumit6597) & [Ambesh Shekhar](https://ambityga.github.io/)



> With so many possibilities of disease and disorders, there had been a surge in demand for medical assistance. Although there are a number of well-practiced physicians and doctors, the inability to reach has always been a concern to many diseased ones. To fill this gap, we propose a model based on deep learning methods, a conversational dialogue system that is able to provide better medication during the need of such and is able to answer any queries related to one’s health. We use the medical assistance dataset scraped from websites containing professional assistance via a conversation system. We also tend to use the trending transformers trained on large datasets, for text generation like BERT, GPT2, and BART. We also performed a comparative study of models and based on the analysis, the results shows that BART model generates a doctor-like response and contains clinically informative data. The overall generated results were very promising and show that pre-trained transformers are reliable for developing automated medical assistance system and doctorlike-treatments.


This repository contains python scripts used for the analysis on a research for EMNLP workshop(ClinicalEMNLP). The research provides a technique to adapt the methods of deep-learning with transfer-learning for a dialogue system which uses transformers like BERT, GPT ,and BART to generate informative dialogues, easy to understand and familiar to human-like conversations. It has been trained on datasets containing interactions between patients and doctors.

## Methods
We have used three main transformers-BERT, GPT, and BART; for the analysis. Architecture and training used with BERT and GPT are same where we have used huggingface-transformers packages, to built the script, whereas we have constructed a wrapper class for BART to get access to the encoder and decoder of the BART, and also a linear layer for linear transformation from the decoder layer. For more information go through my paper.

## Testing
You can test the scripts from this repo on this data. To test the repo, follow the steps:
- Run this command in the command-line/command-prompt,```pip install -r requirements.txt```This will download the required packages neccesary to run the program
- Next go to the Model's(BERT,GPT,BART) directory you want to test.
- First run the ```preprocess.py```, this will generate a json file containing interactions between patient and doctors, and sliced data into training, validation and testing dataset.
- Second, run the ```train.py```, this will update the weights of the model for the required dataset.
- To calculate the perplexity score on each data(train, validation, test) run ```calculate_perplexity.py```. This will calculate the perplexity score for each data based on your updated model's parameters.
- To generate response and calculate BLEU, NIST, DIST, METEOR, Entropy ,and Average length generated run ```generate_text.py```.

## References
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf "BERT")
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf "GPT2")
- [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/pdf/1910.13461 "BART")


