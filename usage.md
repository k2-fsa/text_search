# Usage

This section describes how to compile and use this project.

(For developers only)

```bash
git clone https://github.com/danpovey/text_search
cd text_search

mkdir build
cd build
cmake ..
make -j
make test

# set PYTHONPATH so that you can use "import text_search"

export PYTHONPATH=$PWD/../textsearch/python:$PWD/lib:$PYTHONPATH
```

Now you can use

```bash
python3 -c "import textsearch; print(textsearch.__file__)"
```


**Caution**: We did not use either `python3 setup.py install` or `pip install`.
We only set the environment variable `PYTHONPATH`.


