# Usage & Training Instructions

SSH into GCP VM using VS Code extention (enter {first name}{last name}@{IP Address of VM} for example brendanmclaughlin@34.125.186.157)

'open folder' bert-sim-cse-deployment-vm

make changes and commit to github branches (as you noramlly would)

to run training, use tmux so you can close your computer or even quit terminal session and it will keep running

do tmux new -s {name your session} OR tmux ls to see existing sessions

do Ctrl+b d to detach from session

to reattach do tmux a -t {name of session}

run your training command

do Ctrl+b d to detach from session

# CS 224N Default Final Project 2024 - Multitask BERT

This is the default final project for the Stanford CS 224N class. Please refer to the project handout on the course website for detailed instructions and an overview of the codebase.

This project comprises two parts. In the first part, you will implement some important components of the BERT model to better understand its architecture. 
In the second part, you will use the embeddings produced by your BERT model on three downstream tasks: sentiment classification, paraphrase detection, and semantic similarity. You will implement extensions to improve your model's performance on the three downstream tasks.

In broad strokes, Part 1 of this project targets:
* bert.py: Missing code blocks.
* classifier.py: Missing code blocks.
* optimizer.py: Missing code blocks.

And Part 2 targets:
* multitask_classifier.py: Missing code blocks.
* datasets.py: Possibly useful functions/classes for extensions.
* evaluation.py: Possibly useful functions/classes for extensions.

## Setup instructions

Follow `setup.sh` to properly setup a conda environment and install dependencies.

## Acknowledgement

The BERT implementation part of the project was adapted from the "minbert" assignment developed at Carnegie Mellon University's [CS11-711 Advanced NLP](http://phontron.com/class/anlp2021/index.html),
created by Shuyan Zhou, Zhengbao Jiang, Ritam Dutt, Brendon Boldt, Aditya Veerubhotla, and Graham Neubig.

Parts of the code are from the [`transformers`](https://github.com/huggingface/transformers) library ([Apache License 2.0](./LICENSE)).
