:py:mod:`bocoel.factories.embedders`
====================================

.. py:module:: bocoel.factories.embedders

.. autodoc2-docstring:: bocoel.factories.embedders
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`EmbedderName <bocoel.factories.embedders.EmbedderName>`
     - .. autodoc2-docstring:: bocoel.factories.embedders.EmbedderName
          :summary:

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`embedder <bocoel.factories.embedders.embedder>`
     - .. autodoc2-docstring:: bocoel.factories.embedders.embedder
          :summary:

API
~~~

.. py:class:: EmbedderName()
   :canonical: bocoel.factories.embedders.EmbedderName

   Bases: :py:obj:`bocoel.common.StrEnum`

   .. autodoc2-docstring:: bocoel.factories.embedders.EmbedderName

   .. rubric:: Initialization

   .. autodoc2-docstring:: bocoel.factories.embedders.EmbedderName.__init__

   .. py:attribute:: SBERT
      :canonical: bocoel.factories.embedders.EmbedderName.SBERT
      :value: 'SBERT'

      .. autodoc2-docstring:: bocoel.factories.embedders.EmbedderName.SBERT

   .. py:attribute:: HUGGINGFACE
      :canonical: bocoel.factories.embedders.EmbedderName.HUGGINGFACE
      :value: 'HUGGINGFACE'

      .. autodoc2-docstring:: bocoel.factories.embedders.EmbedderName.HUGGINGFACE

   .. py:attribute:: HUGGINGFACE_ENSEMBLE
      :canonical: bocoel.factories.embedders.EmbedderName.HUGGINGFACE_ENSEMBLE
      :value: 'HUGGINGFACE_ENSEMBLE'

      .. autodoc2-docstring:: bocoel.factories.embedders.EmbedderName.HUGGINGFACE_ENSEMBLE

.. py:function:: embedder(name: str | bocoel.factories.embedders.EmbedderName, /, *, model_name: str | list[str], device: str = 'auto', batch_size: int) -> bocoel.Embedder
   :canonical: bocoel.factories.embedders.embedder

   .. autodoc2-docstring:: bocoel.factories.embedders.embedder
