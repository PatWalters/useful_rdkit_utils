Installation
============
useful_rdkit_utils requires Python 3.11 or newer.

The most recent release can be installed from
`PyPI <https://pypi.org/project/useful_rdkit_utils>`_ with:

.. code-block:: shell

    $ pip install useful_rdkit_utils

The most recent code and data can be installed directly from GitHub with:

.. code-block:: shell

    $ pip install git+https://github.com/PatWalters/useful_rdkit_utils.git

To install in development mode, use the following:

.. code-block:: shell

    $ git clone https://github.com/PatWalters/useful_rdkit_utils.git
    $ cd useful_rdkit_utils
    $ pip install -e .

Extras
------
Everything in the API reference works from a plain install except for the pieces
listed below, which need an optional dependency.

.. list-table::
   :header-rows: 1
   :widths: 15 25 60

   * - Extra
     - Brings in
     - Needed for
   * - ``extras``
     - click
     - the :doc:`build_ring_dictionary <cli>` command line tool
   * - ``jupyter``
     - IPython
     - the ``rd_*`` helpers in :doc:`jupyter`, which configure RDKit's notebook
       rendering. RDKit implements that on top of IPython.
   * - ``all``
     - click, IPython
     - both of the above
   * - ``docs``
     - the Sphinx toolchain
     - building this documentation
   * - ``tests``
     - pytest
     - running the test suite

Install one with, for example:

.. code-block:: shell

    $ pip install useful_rdkit_utils[all]

Running the tests
-----------------
Some of the tests download data from the internet and are marked ``network``.
To run everything else:

.. code-block:: shell

    $ pip install -e .[tests,all]
    $ pytest -m "not network"
