import sphinx_rtd_theme
project = 'Kolmogorov Arnold Network'
copyright = '2024, Ziming Liu'
author = 'Ziming Liu'
extensions = ['sphinx_rtd_theme', 'sphinx.ext.autodoc',
    'sphinx.ext.autosectionlabel']
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']


def skip(app, what, name, obj, would_skip, options):
    if name == '__init__':
        return False
    return would_skip


def setup(app):
    app.connect('autodoc-skip-member', skip)


autodoc_mock_imports = ['numpy', 'torch', 'torch.nn', 'matplotlib',
    'matplotlib.pyplot', 'tqdm', 'sympy', 'scipy', 'sklearn', 'torch.optim']
source_suffix = ['.rst', '.md']
