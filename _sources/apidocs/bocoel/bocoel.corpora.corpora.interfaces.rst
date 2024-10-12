:py:mod:`bocoel.corpora.corpora.interfaces`
===========================================

.. py:module:: bocoel.corpora.corpora.interfaces

.. autodoc2-docstring:: bocoel.corpora.corpora.interfaces
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`Corpus <bocoel.corpora.corpora.interfaces.Corpus>`
     - .. autodoc2-docstring:: bocoel.corpora.corpora.interfaces.Corpus
          :summary:

API
~~~

.. py:class:: Corpus
   :canonical: bocoel.corpora.corpora.interfaces.Corpus

   Bases: :py:obj:`typing.Protocol`

   .. autodoc2-docstring:: bocoel.corpora.corpora.interfaces.Corpus

   .. py:attribute:: storage
      :canonical: bocoel.corpora.corpora.interfaces.Corpus.storage
      :type: bocoel.corpora.storages.Storage
      :value: None

      .. autodoc2-docstring:: bocoel.corpora.corpora.interfaces.Corpus.storage

   .. py:attribute:: index
      :canonical: bocoel.corpora.corpora.interfaces.Corpus.index
      :type: bocoel.corpora.indices.Index
      :value: None

      .. autodoc2-docstring:: bocoel.corpora.corpora.interfaces.Corpus.index

   .. py:method:: __repr__() -> str
      :canonical: bocoel.corpora.corpora.interfaces.Corpus.__repr__
