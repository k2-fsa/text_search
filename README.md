# Introduction

The text_search project can be used to create ASR (automatic speech recognition) dataset with long-form audios and even longer texts. 

The core of text_search is a general audio alignment pipeline, which aims to align the audio files to the corresponding text and split them into short segments, while also excluding segments of audio that do not correspond exactly with the aligned text.


# Installation

## With pip

```
pip install fasttextsearch
```

## For developers

```bash
pip install numpy

git clone https://github.com/danpovey/text_search
cd text_search

mkdir build
cd build
cmake ..
make -j
make test

# set PYTHONPATH so that you can use "import textsearch"

export PYTHONPATH=$PWD/../textsearch/python:$PWD/lib:$PYTHONPATH
```

Now you can use

```bash
python3 -c "import textsearch; print(textsearch.__file__)"
```


**Caution**: We did not use either `python3 setup.py install` or `pip install`.
We only set the environment variable `PYTHONPATH`.



# Recipes

- [libriheavy](examples/libriheavy)
- [subtitle](examples/subtitle)


# References
More explainations are available in the following paper:

```
@misc{kang2023libriheavy,
      title={Libriheavy: a 50,000 hours ASR corpus with punctuation casing and context}, 
      author={Wei Kang and Xiaoyu Yang and Zengwei Yao and Fangjun Kuang and Yifan Yang and Liyong Guo and Long Lin and Daniel Povey},
      year={2023},
      eprint={2309.08105},
      archivePrefix={arXiv},
      primaryClass={eess.AS}
}
```