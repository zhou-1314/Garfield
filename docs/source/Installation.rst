Installation
============

Stable version
~~~~~~~~~~~~~~

To use Garfield, first install it:

**Recommended**: install *Garfield* in a new virtual environment::

    conda create -n env_garfield python=3.9
    conda activate env_garfield

**Additional Libraries**

To use Garfield, you need to install some external libraries. These include:
 - [PyTorch]
 - [PyTorch Scatter]
 - [PyTorch Sparse]
 - [bedtools]

We recommend to install the PyTorch libraries with GPU support. If you have
CUDA, this can be done as::

    pip install torch==${TORCH}
    pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html

where `${TORCH}` and `${CUDA}` should be replaced by the specific PyTorch and
CUDA versions, respectively.

For example, for PyTorch 2.0.0 and CUDA 11.7, type::

    pip install torch==2.0.0
    pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.0.0+cu117.html


To install bedtools, you can use conda::

    conda install bedtools=2.29.0 -c bioconda


Finally, to install Garfield with pip, run::

    pip install Garfield

Dev version
~~~~~~~~~~~

To install the development version on `GitHub <https://github.com/zhou-1314/Garfield/>`_,
first install torch::

    pip install torch torchvision torchaudio

then run::

    git clone https://github.com/zhou-1314/Garfield.git
    cd Garfield
    python setup.py install

or::

    pip install git+https://github.com/zhou-1314/Garfield.git

After a correct installation, you should be able to import the module without errors::

    import Garfield as gf
