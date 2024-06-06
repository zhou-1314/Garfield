Installation
============

Stable version
~~~~~~~~

To use Garfield, first install it:

**Recommended**: install *Garfield* in a new virtual environment::

    conda create -n env_garfield python=3.9
    conda activate env_garfield

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

