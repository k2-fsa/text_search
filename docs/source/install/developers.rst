For developers
==============

Please use the following commands to install `fasttextsearch`_:

.. code-block:: bash

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

To test the you have installed `fasttextsearch`_ successfully, please run:

.. code-block:: bash

  python3 -c "import textsearch; print(textsearch.__file__)"

It should print something like below:

.. code-block:: bash

   /Users/fangjun/open-source/text_search/textsearch/python/textsearch/__init__.py


