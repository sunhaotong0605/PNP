# Prediction of soil probiotics based on foundation model representation enhancement and stacked aggregation classifier

## Introduction
We utilize genomic foundation models to generate representations from samples’ sequences, and then, enhance them by deeply integrating domain-specific engineered features. The enhanced representations enable training a powerful classifier for a target task. We also design a stacked aggregation classifier. It predicts the label of a sample with only leveraging partial sequence segments from this sample, effectively addressing the challenges in processing long sequences. The proposed method is applied on prediction of soil probiotics.

## Schematic Diagram
<div style="text-align: center;">
    <img src="img/fig1.jpg" alt="fig1" width="493" height="725">
</div>
Figure 1. Overview of the proposed method. The genomic sequence of a bacterial sample is divided into segments. Its partial segments are input into a pre-trained foundation model to generate representations, and engineered features are extracted from these segments. The foundation model representation and engineered feature vectors are aligned, and then, the foundation model representations are enhanced by deeply integrating the engineered features. The enhanced representations are fed into the stacked aggregation classifier. The first-level classifier processes each enhanced representation to obtain a score. All scores are aggregated into a vector, which is input into the second-level classifier to output the final label and score.

## Quick Start

### Download the GitHub Repository
[Download](https://github.com/sunhaotong0605/SPP_FMRESAC/archive/refs/heads/main.zip) this GitHub repository, and extract the contents into a folder.

### Data Description
The proposed method requires data in [FASTA](https://www.ncbi.nlm.nih.gov/genbank/fastaformat/) format as input. All data used will be made public as a supplementary table after the paper is accepted.

## Install
```bash
# Python environment constructed by Conda
conda create -n SPP_FMRESAC python=3.8.15
conda activate SPP_FMRESAC
git clone https://github.com/sunhaotong0605/SPP_FMRESAC.git
cd SPP_FMRESAC
pip install -r requirements.txt
```

## Usage
#### Foundation model weights.
Download the weights of [Nucleotide Transformer-50M](https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-50m-multi-species) to the model_weights/NT_50M/ folder.

Download the weights of [EVO-7B](https://huggingface.co/togethercomputer/evo-1-8k-base) to the model_weights/EVO_7B/ folder.

#### Prediction for target samples.
```bash
python main.py model_name=xxx input_path=xxx output_path=xxx
```
--model_name: A selected foundation model for generating representations, and the candidates only can be "NT_50M" or "EVO_7B".

--input_path: A path of FASTA files, where each FASTA file is a target sample.

--output_path: A path for outputting files.

Example: python main.py model_name=NT_50M input_path=/home/sample/input_path/ output_path=/home/result/

The input path can contain one or multiple FASTA files (samples).

For each sample, the output contains

├── Sequence_segments: sequence segments

├── Engineered_features: engineered features (.pkl)

├── Foundation_model_representations: foundation model representations (.pkl)

├── Enhanced_representations: enhanced representations (.pkl)

├── Prediction_results: predicted labels and confidence scores (.txt)

└── Temp: other required files

If a sample's sequence has been segmented, i.e., the Sequence_segments folder has been existed in the output path, sequence segmentation step will be skipped, and existing sequence segments are directly used for prediction. 

#### notice
Each prediction involves randomly selecting partial segments from a sample, may result in inconsistent outputs across multiple runs due to differences in the selected segments sets.

## License
MIT License. See [LICENSE](LICENSE.txt) for details.

## Citation
Kang Q, Sun H, Wang Y, et al. Prediction of soil probiotics based on foundation model representation enhancement and stacked aggregation classifier. bioRxiv. doi:
