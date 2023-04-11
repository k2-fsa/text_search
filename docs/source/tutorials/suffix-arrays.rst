Suffix arrays
=============

Please refer to `Wikipedia <https://en.wikipedia.org/wiki/Suffix_array>`_
for what suffix array is.

This sections describes how to use Python APIs from `fastsearch`_
to create suffix arrays.

.. caution::

    The above `Wikipedia <https://en.wikipedia.org/wiki/Suffix_array>`_ article
    assumes the  sentinel letter ``$`` is lexicographically smaller than any other
    character, while in `fastsearch`_ we assumes it is larger than all other characters
    and we call the sentinel letter as ``EOS``.
