# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import re
import sys

import sphinx_rtd_theme

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../../textsearch/python"))
sys.path.insert(0, os.path.abspath("../../build/lib"))


# -- Project information -----------------------------------------------------

project = "fasttextsearch"
copyright = "2023, The Next-gen Kaldi Development Team"
author = "The Next-gen Kaldi Development Team"


def get_version():
    cmake_file = "../../CMakeLists.txt"
    with open(cmake_file) as f:
        content = f.read()

    version = re.search(r"set\(FTS_VERSION (.*)\)", content).group(1)
    return version.strip('"')


# The full version, including alpha/beta/rc tags
version = get_version()
release = version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "recommonmark",
    "sphinx.ext.autodoc",
    "sphinx.ext.githubpages",
    "sphinx.ext.linkcode",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx_rtd_theme",
    "sphinx_tabs.tabs",
    "sphinxcontrib.youtube",
    "sphinx.ext.mathjax",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []
master_doc = "index"


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_show_sourcelink = True

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

pygments_style = "sphinx"
numfig = True

html_context = {
    "display_github": True,
    "github_user": "danpovey",
    "github_repo": "text_search",
    "github_version": "master",
    "conf_py_path": "/docs/source/",
}

# refer to
# https://sphinx-rtd-theme.readthedocs.io/en/latest/configuring.html
html_theme_options = {
    "logo_only": False,
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": True,
}
rst_epilog = """
.. _k2-fsa: https://github.com/k2-fsa
.. _fasttextsearch: https://github.com/danpovey/text_search
"""

mathjax_path = (
    "https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
)

# Resolve function for the linkcode extension.
# Modified from https://github.com/rwth-i6/returnn/blob/master/docs/conf.py
def linkcode_resolve(domain, info):
    def find_source():
        # try to find the file and line number, based on code from numpy:
        # https://github.com/numpy/numpy/blob/master/doc/source/conf.py#L286
        obj = sys.modules[info["module"]]
        for part in info["fullname"].split("."):
            obj = getattr(obj, part)
        import inspect
        import os

        fn = inspect.getsourcefile(obj)
        print("fn", fn)
        fn = os.path.relpath(fn, start="text_search")
        print("fn2", fn)
        source, lineno = inspect.getsourcelines(obj)
        return fn, lineno, lineno + len(source) - 1

    if domain != "py" or not info["module"]:
        return None
    try:
        filename = "{}#L{}-L{}".format(*find_source())
    except Exception:
        return None

    print("filename", filename)
    idx = filename.rfind("textsearch")
    filename = filename[idx:]
    return f"https://github.com/danpovey/text_search/blob/master/textsearch/python/{filename}"
