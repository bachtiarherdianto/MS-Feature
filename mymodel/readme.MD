This directory contains the trained and serialized (`.pkl`) machine learning classification models used in our study. The models were trained to learn patterns from instance-level and solution-level features, as described in the paper.

All trained models are provided as serialized pickle (`.pkl`) files to ensure reproducibility of the experimental results.

> **Note:** Due to file size limitations, the trained model files are not included directly in this repository. They can be downloaded from **[Google Drive](https://drive.google.com/drive/folders/1XV8x7FYPh2et7MIqZwt0pelGZTM3tmlE?usp=sharing)**. After downloading, please place the files in the `mymodel/` directory before running the code.

The models are automatically loaded and evaluated within `main.ipynb`. No additional training is required to reproduce the results reported in the paper.

---