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
import sys
import sphinx_rtd_theme
import inspect
from os.path import relpath, dirname
sys.path.insert(0, os.path.abspath('../src/pgbm_nb/'))
sys.path.insert(0, os.path.abspath('../src/pgbm/'))

# -- Project information -----------------------------------------------------

project = 'PGBM'
copyright = '2021, Olivier Sprangers, AirLab'
author = 'Olivier Sprangers'

# The full version, including alpha/beta/rc tags
release = '1.5'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
# extensions = ['myst_parser', 'sphinx_rtd_theme', 'sphinx.ext.autodoc', 'sphinx.ext.viewcode', 'sphinx.ext.linkcode']
extensions = ['myst_parser', 'sphinx_rtd_theme', 'sphinx.ext.autodoc', 'sphinx.ext.linkcode']

# From numpy: https://github.com/numpy/numpy/blob/83828f52b287fefb3d8753a21bd3441997a4d687/doc/source/conf.py#L303-L348
import pgbm 

def linkcode_resolve(domain, info):
    """
    Determine the URL corresponding to Python object
    """
    if domain != 'py':
        return None

    modname = info['module']
    fullname = info['fullname']

    submod = sys.modules.get(modname)
    if submod is None:
        return None
    
    obj = submod
    for part in fullname.split('.'):
        try:
            obj = getattr(obj, part)
        except Exception:
            return None

    try:
        fn = inspect.getsourcefile(obj)
    except Exception:
        fn = None
    if not fn:
        return None

    try:
        source, lineno = inspect.getsourcelines(obj)
    except Exception:
        lineno = None

    if lineno:
        linespec = "#L%d-L%d" % (lineno, lineno + len(source) - 1)
    else:
        linespec = ""

    fn = relpath(fn, start=dirname(dirname(pgbm.__file__)))
    return "https://github.com/elephaint/pgbm/blob/main/src/%s%s" % (fn, linespec)

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']