:py:mod:`bocoel.factories.storages`
===================================

.. py:module:: bocoel.factories.storages

.. autodoc2-docstring:: bocoel.factories.storages
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`StorageName <bocoel.factories.storages.StorageName>`
     - .. autodoc2-docstring:: bocoel.factories.storages.StorageName
          :summary:

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`storage <bocoel.factories.storages.storage>`
     - .. autodoc2-docstring:: bocoel.factories.storages.storage
          :summary:

API
~~~

.. py:class:: StorageName()
   :canonical: bocoel.factories.storages.StorageName

   Bases: :py:obj:`bocoel.common.StrEnum`

   .. autodoc2-docstring:: bocoel.factories.storages.StorageName

   .. rubric:: Initialization

   .. autodoc2-docstring:: bocoel.factories.storages.StorageName.__init__

   .. py:attribute:: PANDAS
      :canonical: bocoel.factories.storages.StorageName.PANDAS
      :value: 'PANDAS'

      .. autodoc2-docstring:: bocoel.factories.storages.StorageName.PANDAS

   .. py:attribute:: DATASETS
      :canonical: bocoel.factories.storages.StorageName.DATASETS
      :value: 'DATASETS'

      .. autodoc2-docstring:: bocoel.factories.storages.StorageName.DATASETS

.. py:function:: storage(storage: str | bocoel.factories.storages.StorageName, /, *, path: str = '', name: str = '', split: str = '') -> bocoel.Storage
   :canonical: bocoel.factories.storages.storage

   .. autodoc2-docstring:: bocoel.factories.storages.storage
