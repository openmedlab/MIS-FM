# Medical Image Segmentation Foundation Model
<!-- select Model and/or Data and/or Code as needed>
### Welcome to OpenMEDLab! 👋

<!--
**Here are some ideas to get you started:**
🙋‍♀️ A short introduction - what is your organization all about?
🌈 Contribution guidelines - how can the community get involved?
👩‍💻 Useful resources - where can the community find your docs? Is there anything else the community should know?
🍿 Fun facts - what does your team eat for breakfast?
🧙 Remember, you can do mighty things with the power of [Markdown](https://docs.github.com/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)
-->


<!-- Insert the project banner here -->
<div align="center">
    <a href="https://"><img width="1000px" height="auto" src="https://github.com/openmedlab/sampleProject/blob/main/banner_sample.png"></a>
</div>

---

<!-- Select some of the point info, feel free to delete -->
[![Twitter](https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Ftwitter.com%2Fopendilab)](https://twitter.com/opendilab)
[![PyPI](https://img.shields.io/pypi/v/DI-engine)](https://pypi.org/project/DI-engine/)
![Conda](https://anaconda.org/opendilab/di-engine/badges/version.svg)
![Conda update](https://anaconda.org/opendilab/di-engine/badges/latest_release_date.svg)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/DI-engine)
![PyTorch Version](https://img.shields.io/badge/dynamic/json?color=blue&label=pytorch&query=%24.pytorchVersion&url=https%3A%2F%2Fgist.githubusercontent.com/PaParaZz1/54c5c44eeb94734e276b2ed5770eba8d/raw/85b94a54933a9369f8843cc2cea3546152a75661/badges.json)


![Loc](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/HansBug/3690cccd811e4c5f771075c2f785c7bb/raw/loc.json)
![Comments](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/HansBug/3690cccd811e4c5f771075c2f785c7bb/raw/comments.json)

![Style](https://github.com/opendilab/DI-engine/actions/workflows/style.yml/badge.svg)
![Docs](https://github.com/opendilab/DI-engine/actions/workflows/doc.yml/badge.svg)
![Unittest](https://github.com/opendilab/DI-engine/actions/workflows/unit_test.yml/badge.svg)
![Algotest](https://github.com/opendilab/DI-engine/actions/workflows/algo_test.yml/badge.svg)
![deploy](https://github.com/opendilab/DI-engine/actions/workflows/deploy.yml/badge.svg)
[![codecov](https://codecov.io/gh/opendilab/DI-engine/branch/main/graph/badge.svg?token=B0Q15JI301)](https://codecov.io/gh/opendilab/DI-engine)

![GitHub Org's stars](https://img.shields.io/github/stars/opendilab)
[![GitHub stars](https://img.shields.io/github/stars/opendilab/DI-engine)](https://github.com/opendilab/DI-engine/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/opendilab/DI-engine)](https://github.com/opendilab/DI-engine/network)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/opendilab/DI-engine)
[![GitHub issues](https://img.shields.io/github/issues/opendilab/DI-engine)](https://github.com/opendilab/DI-engine/issues)
[![GitHub pulls](https://img.shields.io/github/issues-pr/opendilab/DI-engine)](https://github.com/opendilab/DI-engine/pulls)
[![Contributors](https://img.shields.io/github/contributors/opendilab/DI-engine)](https://github.com/opendilab/DI-engine/graphs/contributors)
[![GitHub license](https://img.shields.io/github/license/opendilab/DI-engine)](https://github.com/opendilab/DI-engine/blob/master/LICENSE)

This repository provides the official implementation of "MIS-FM: Medical Image Segmentation Foundation Model 
Pretrained with Large-Scale Unannotated 3D Images using Volume Fusion".

## Key Features

- A new self-supervised learning method based on Volume Fusion that is a segmentation-based pretext task.
- A new network architecture PCT-Net that combines the advantages of CNNs and Transformers.
- A foundation model that is trained from 100k unannotated 3D CT scans. 

## Links

- [Paper](https://)
- [Model](https://)
- [Code](https://) 
<!-- [Code] may link to your project at your institute>


<!-- give a introduction of your project -->
## Details

The following figure shows an overview of our proposed method for pretraining with unannotated 3D medical images. We introduce a pretext task based on sudo-segmentation, where Volume Fusion is used  to generate paired images and segmentation labels to pretrain the 3D segmentation model, which can better match the downstream task of segmentation than existing Self-Supervised Learning (SSL) methods. 

<!-- Insert a pipeline of your algorithm here if got one -->
<div align="center">
    <a href="https://"><img width="500px" height="auto" src="figures/framework.png"></a>
</div>

The pretraining strategy is combined with our proposed PCT-Net to obtain a pretrained model that is applied to segmentation of different objects from 3D medical images after fine tuning with a small set of labeled data.


## Datasets

<div align="center">
    <a href="https://"><img width="500px" height="auto" src="figures/datasets.png"></a>
</div>

## Get Started

**Main Requirements**  
> connected-components-3d  
> h5py==3.6.0  
> monai==0.9.0  
> torch==1.11.0  
> tqdm  
> fastremap  

**Installation**
```bash
pip install DDD
```

**Download Model**


**Preprocess**
```bash
python DDD
```


**Training**
```bash
python DDD
```


**Validation**
```bash
python DDD
```


**Testing**
```bash
python DDD
```

## 🙋‍♀️ Feedback and Contact

- Email
- Webpage 
- Social media


## 🛡️ License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgement

A lot of code is modified from [monai](https://github.com/Project-MONAI/MONAI).

## 📝 Citation

If you find this repository useful, please consider citing this paper:
```
@article{John2023,
  title={paper},
  author={John},
  journal={arXiv preprint arXiv:},
  year={2023}
}
```

