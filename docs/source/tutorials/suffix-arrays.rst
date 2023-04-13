Suffix arrays
=============

Please refer to `Wikipedia <https://en.wikipedia.org/wiki/Suffix_array>`_
for what suffix array is.

This sections describes how to use Python APIs from `fasttextsearch`_
to create suffix arrays.

.. caution::

    The above `Wikipedia <https://en.wikipedia.org/wiki/Suffix_array>`_ article
    assumes the  sentinel letter ``$`` is lexicographically smaller than any other
    character, while in `fasttextsearch`_ we assume it is larger than all other characters
    and we call the sentinel letter ``EOS``.

The following code shows how to create the suffix array for the string ``banana``:

.. literalinclude:: ./code/suffix-array.py
   :language: python
   :caption: Create suffix array for the string ``banana``.

Different from the example in `<https://en.wikipedia.org/wiki/Suffix_array>`_,
``$`` is the largest symbol in `fasttextsearch`_, so the first smallest
substring is ``anana$`` instead of ``$``.

Please see :func:`textsearch.create_suffix_array` for details.
