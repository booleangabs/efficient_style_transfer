# Efficient Style Transfer Inference

<p align="center">
  <img src="assets/sample_result.jpeg" alt="Banner" width="80%">
</p>

## Description

This project is being conducted as part of the Artificial Intelligence Course (IF1017) at the [Center of Informatics - Federal University of Pernambuco](https://portal.cin.ufpe.br/). The team is composed of [Gabriel Tavares](https://github.com/booleangabs), [Bruno Carvalho](https://github.com/Bruno0205) and [Gabriel Vasconcelos](https://github.com/GabrielVasconcel). Our goal is to build and fully functional web app capable of running a single-pass method of image style transfer. Initially, the style images are fixed and can be selected from a drop-down list in the app (each style has a model trained for it). In the future, support for style as input may be added. The installation instructions are included below but a quick access to the project through the HuggingFace Spaces interface will be added later on.

## Installing dependencies
This step depends on an [Anaconda](https://www.anaconda.com/download) or [Miniconda](/docs/getting-started/miniconda/main) distribution.

`conda env create -f environment.yml`

`conda activate efficient_style_transfer`

## Running

In a terminal window with the enviroment activated, run the following command to start the server:
`python src/app.py`

You may use one of the examples as reference on how to send requests to the server. A sample image is provided for quick testing.