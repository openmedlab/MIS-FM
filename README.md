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


<!-- Insert the project banner here 
<div align="center">
    <a href="https://"><img width="1000px" height="auto" src="https://github.com/openmedlab/sampleProject/blob/main/banner_sample.png"></a>
</div>
-->

---

<!-- Select some of the point info, feel free to delete -->
<!-- [![Twitter](https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Ftwitter.com%2Fopendilab)](https://twitter.com/opendilab) -->
[![PyPI](https://img.shields.io/pypi/v/DI-engine)](https://pypi.org/project/DI-engine/)
![Conda](https://anaconda.org/opendilab/di-engine/badges/version.svg)
![Conda update](https://anaconda.org/opendilab/di-engine/badges/latest_release_date.svg)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/DI-engine)
![PyTorch Version](https://img.shields.io/badge/dynamic/json?color=blue&label=pytorch&query=%24.pytorchVersion&url=https%3A%2F%2Fgist.githubusercontent.com/PaParaZz1/54c5c44eeb94734e276b2ed5770eba8d/raw/85b94a54933a9369f8843cc2cea3546152a75661/badges.json)


<!-- ![Loc](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/HansBug/3690cccd811e4c5f771075c2f785c7bb/raw/loc.json)
![Comments](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/HansBug/3690cccd811e4c5f771075c2f785c7bb/raw/comments.json)

![Style](https://github.com/opendilab/DI-engine/actions/workflows/style.yml/badge.svg)
![Docs](https://github.com/opendilab/DI-engine/actions/workflows/doc.yml/badge.svg)
![Unittest](https://github.com/opendilab/DI-engine/actions/workflows/unit_test.yml/badge.svg)
![Algotest](https://github.com/opendilab/DI-engine/actions/workflows/algo_test.yml/badge.svg)
![deploy](https://github.com/opendilab/DI-engine/actions/workflows/deploy.yml/badge.svg)
[![codecov](https://codecov.io/gh/opendilab/DI-engine/branch/main/graph/badge.svg?token=B0Q15JI301)](https://codecov.io/gh/opendilab/DI-engine) -->

<!-- ![GitHub Org's stars](https://img.shields.io/github/stars/opendilab)
[![GitHub stars](https://img.shields.io/github/stars/opendilab/DI-engine)](https://github.com/opendilab/DI-engine/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/opendilab/DI-engine)](https://github.com/opendilab/DI-engine/network)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/opendilab/DI-engine)
[![GitHub issues](https://img.shields.io/github/issues/opendilab/DI-engine)](https://github.com/opendilab/DI-engine/issues)
[![GitHub pulls](https://img.shields.io/github/issues-pr/opendilab/DI-engine)](https://github.com/opendilab/DI-engine/pulls)
[![Contributors](https://img.shields.io/github/contributors/opendilab/DI-engine)](https://github.com/opendilab/DI-engine/graphs/contributors) -->
[![GitHub license](https://img.shields.io/github/license/opendilab/DI-engine)](https://github.com/opendilab/DI-engine/blob/master/LICENSE)

This repository provides the official implementation of "MIS-FM: Medical Image Segmentation Foundation Model 
Pretrained with Large-Scale Unannotated 3D Images using Volume Fusion".

## Key Features

- A new self-supervised learning method based on Volume Fusion that is a segmentation-based pretext task.
- A new network architecture PCT-Net that combines the advantages of CNNs and Transformers.
- A foundation model that is trained from 100k unannotated 3D CT scans. 

## Links

- [Paper (To be shown on arxiv soon)](https://)
- [Model (Google Drive)](https://https://drive.google.com/file/d/1jQc-2hhsp3EyZj54_KEJte85diUtW8Fg/view?usp=sharing)
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

We used 10k CT volumes from public datasets and 98k private CT volumes for pretraining.
<div align="center">
    <a href="https://"><img width="500px" height="auto" src="figures/datasets.png"></a>
</div>

## Demo for using the pretrained model

**Main Requirements**  
> torch==1.10.2  
> PyMIC 

To use [PyMIC](https://github.com/HiLab-git/PyMIC), please download the latest code in the master branch, and add the path of PyMIC source code to `PYTHONPATH` environmental variable. See `bash.sh` for example.


**Demo data**

In this demo, we show the use of PCT-Net for left atrial segmentation. The dataset can be downloaded from [PYMIC_data](https://drive.google.com/file/d/1eZakSEBr_zfIHFTAc96OFJix8cUBf-KR/view?usp=sharing).

The dataset, network and training/testing settings can be found in configuration files: `demo/pctnet_scratch.cfg` and `demo/pctnet_pretrain.cfg` for training from scratch and using the pretrained weights, respectively.

After downloading the data, edit the value of `root_dir` in the configuration files, and make sure the path to the images is correct.

**Training**
```bash
python train.py demo/pctnet_scratch.cfg
```
or 

```bash
python train.py demo/pctnet_pretrain.cfg
```

**Inference**
```bash
python predict.py demo/pctnet_scratch.cfg
```
or 

```bash
python predict.py demo/pctnet_pretrain.cfg
```

**Evaluation**
```bash
python $PyMIC_path/pymic/util/evaluation_seg.py -cfg demo/evaluation.cfg
```
You may need to edit `demo/evaluation.cfg` to specify the path of segmentation results before evaluating the performance.

In this simple demo, the segmentation Dice was 90.71% and 92.73% for training from scratch and from the pretrained weights, respectively.

## 🛡️ License

This project is under the Apache license. See [LICENSE](LICENSE) for details.

<!-- ## 📝 Citation

If you find this repository useful, please consider citing this paper:
```
@article{John2023,
  title={paper},
  author={John},
  journal={arXiv preprint arXiv:},
  year={2023}
}
``` -->

