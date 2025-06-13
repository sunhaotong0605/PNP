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
#### Prediction for multiple samples.
```bash
python main_m.py -m model_name -i input_file -o output_path
```
--model_name: A selected foundation model for generating representations, and the candidates only can be "NT_50M", "DNABERT2_117M" or "EVO_7B".

--input_file: An input FASTA file.

--output_path: A path for outputting files.

The input FASTA file can contain one or multiple samples. For each sample, an output folder named after the sample is generated, containing: a directory of sequence segments, foundation model representations (.pkl), engineered features (.pkl), and enhanced representations (.pkl). The predicted labels and confidence scores are printed to the console.

#### Prediction for a sample after sequence segmentation.
```bash
python main_o.py -m model_name -s segment_path -o output_path
```
--model_name: A selected foundation model for generating representations, and the candidates only can be "NT_50M", "DNABERT2_117M" or "EVO_7B".

--segment_path: A path of sequence segments.

--output_path: A path for outputting files.

If a sample's sequence has been segmented, sequence segmentation step can be skipped, and existing sequence segments can be directly used for prediction. This script does not support multi-sample prediction. An output folder named after the sample is generated, containing: foundation model representations (.pkl), engineered features (.pkl), and enhanced representations (.pkl). The predicted label and confidence score are printed to the console.

#### notice
Each prediction involves randomly selecting partial segments from a sample, may result in inconsistent outputs across multiple runs due to differences in the selected segments sets.

## License
MIT License. See [LICENSE](LICENSE.txt) for details.

## Citation
Kang Q, Sun H, Wang Y, et al. Prediction of soil probiotics based on foundation model representation enhancement and stacked aggregation classifier. bioRxiv. doi:
