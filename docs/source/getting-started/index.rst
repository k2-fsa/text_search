Getting started
===============

About
-----

The text_search project can be used to create ASR (automatic speech recognition) dataset with long-form audios and even longer texts. 

The core of text_search is a general audio alignment pipeline, which aims to align the audio files to the corresponding text and split them into short segments, while also excluding segments of audio that do not correspond exactly with the aligned text.

Installation
------------

With pip
********

.. code-block:: bash
  
  pip install fasttextsearch


For developers
**************

Please use the following commands to install `fasttextsearch`_:

.. code-block:: bash

  pip install numpy

  git clone https://github.com/k2-fsa/text_search
  cd text_search

  mkdir build
  cd build
  cmake ..
  make -j
  make test

  # set PYTHONPATH so that you can use "import textsearch"

  export PYTHONPATH=$PWD/../textsearch/python:$PWD/lib:$PYTHONPATH

To test the you have installed `fasttextsearch`_ successfully, please run:

.. code-block:: bash

  python3 -c "import textsearch; print(textsearch.__file__)"

It should print something like below:

.. code-block:: bash

   /Users/fangjun/open-source/text_search/textsearch/python/textsearch/__init__.py

.. hint::
   We did not use either `python3 setup.py install` or `pip install`.
   We only set the environment variable `PYTHONPATH`.

