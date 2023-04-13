Find close matches
==================


Assuming the suffix array was created from a text where the first
``query_len`` positions represent the query text and the remaining positions
represent the reference text, :func:`textsearch.find_close_matches`
returns a list indicating, for each suffix position in the query text, the two
suffix positions in the reference text that immediately precede and follow it lexicographically.

The following gives an example about :func:`textsearch.find_close_matches`.

.. literalinclude:: ./code/find-close-matches.py

We have the query string ``hi`` and the document string ``howareyou``.


For the first character ``h`` from the query, we can see that the first
substring preceding it from the document is ``eyou`` at position 7 in the
``full_text`` and the first substring following it is ``howareyou`` at
position 2 in the ``full_text``, so the close match for ``h`` is ``(7, 2)``.

Similarly, for the second character ``i`` from the query, we can see that the first
substring preceding it from the document is ``howareyou`` at position 2 in the
``full_text`` and the first substring following it is ``ou`` at
position 9 in the ``full_text``, so the close match for ``h`` is ``(2, 9)``.



