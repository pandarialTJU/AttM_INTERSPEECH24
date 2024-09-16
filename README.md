# AttM_INTERSPEECH24

✨ **Official Release Code for the Paper** ✨  
*Attentive Merging of Hidden Embeddings from Pre-trained Speech Model for Anti-spoofing Detection*

## Overview

This repository contains the code for the paper **"Attentive Merging of Hidden Embeddings from Pre-trained Speech Model for Anti-spoofing Detection."** The code is designed to train and evaluate models for detecting spoofed speech using the ASVspoof2019 LA dataset or similar datasets.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Meta Files Preparation](#meta-files-preparation)
- [Running the Code](#running-the-code)
- [Citation](#citation)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.x
- Required Python libraries (listed in `requirements.txt`)

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Meta Files Preparation
Before training or evaluating the model, you need to prepare the meta files (.scp and .tsv files) as follows:

- Example of wav_nosox.scp

```bash
file_name  path/to/filename.flac
LA_T_1002656 /data/asvspoof2019/LA/ASVspoof2019_LA_train/flac/LA_T_1002656.flac
LA_T_1003665 /data/asvspoof2019/LA/ASVspoof2019_LA_train/flac/LA_T_1003665.flac
```
- Example of train.tsv, validation.tsv, and evaluation.tsv, which are saved in a meta directory

```bash
file_name   label
LA_T_1002656    spoof
LA_T_1003665    bonafide

```

## Running the code
To train or evaluate the model, use the following command:
```bash
python -u main_antispoof.py --config ./antispoof.conf \
  --meta_dir /meta/path/to/save/tsv_files \
  --feat_file /feat/path/to/save/wav_nosox.scp \
  --output_dir output/path --seed 2048
```

## Citation
```bash
@inproceedings{pan24c_interspeech,
  title     = {Attentive Merging of Hidden Embeddings from Pre-trained Speech Model for Anti-spoofing Detection},
  author    = {Zihan Pan and Tianchi Liu and Hardik B. Sailor and Qiongqiong Wang},
  year      = {2024},
  booktitle = {Interspeech 2024},
  pages     = {2090--2094},
  doi       = {10.21437/Interspeech.2024-1472},
  issn      = {2958-1796},
}
}
```
```bash
Pan, Z., Liu, T., Sailor, H.B., Wang, Q. (2024) Attentive Merging of Hidden Embeddings from Pre-trained Speech Model for Anti-spoofing Detection. Proc. Interspeech 2024, 2090-2094, doi: 10.21437/Interspeech.2024-1472
```

## Contributing
Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (git checkout -b feature-branch).
3. Commit your changes (git commit -m 'Add a new feature').
4. Push to the branch (git push origin feature-branch).
5. Open a Pull Request.

## License
This project is licensed under the MIT License.

## Contact
For questions or suggestions, please reach out to talpanzh@gmail.com


