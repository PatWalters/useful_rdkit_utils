Command Line Tools
==================
Installing the package puts two commands on your ``PATH``.

build_ring_dictionary
---------------------
Builds a ring-system frequency dictionary from a ChEMBL chemreps file, or looks
up the ring systems in a set of molecules against an existing dictionary. This is
the command line front end to
:func:`~useful_rdkit_utils.ring_systems.create_ring_dictionary` and
:func:`~useful_rdkit_utils.ring_systems.test_ring_system_lookup`.

The command is built with `click <https://click.palletsprojects.com>`_, which is
an optional dependency. Install it with:

.. code-block:: shell

    $ pip install useful_rdkit_utils[extras]

.. code-block:: text

    Usage: build_ring_dictionary [OPTIONS]

    Options:
      --mode TEXT     [build|search]
      --infile TEXT   input file
      --outfile TEXT  output file
      --help          Show this message and exit.

In ``build`` mode ``--infile`` is a ChEMBL chemreps file and ``--outfile`` is the
ring dictionary CSV to write:

.. code-block:: shell

    $ build_ring_dictionary --mode build --infile chembl_35_chemreps.txt --outfile chembl_ring_systems.csv

In ``search`` mode ``--infile`` is a SMILES file and ``--outfile`` is a CSV of the
ring systems found in it, annotated with their frequency in the reference
dictionary. Any option you leave off is prompted for.

silly_walks
-----------
Builds the Morgan fingerprint count dictionary that
:class:`~useful_rdkit_utils.silly_walks.SillyWalks` scores against, from the
``canonical_smiles`` column of a ChEMBL chemreps file. The dictionary is written
as JSON and can be read back with
:meth:`~useful_rdkit_utils.silly_walks.SillyWalks.load_json_dict`.

.. code-block:: text

    usage: silly_walks [-h] -in INPUT_FILE -out OUTPUT_FILE

    Generate count_dict from ChEMBL chemreps file.

    options:
      -h, --help            show this help message and exit
      -in, --input_file INPUT_FILE
                            Name of the input file containing the ChEMBL chemreps data.
      -out, --output_file OUTPUT_FILE
                            Name of the output file where the count_dict will be saved.

For example:

.. code-block:: shell

    $ silly_walks -in chembl_35_chemreps.txt -out chembl_35_counts.json
