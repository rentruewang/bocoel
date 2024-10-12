:py:mod:`bocoel.core.exams.managers`
====================================

.. py:module:: bocoel.core.exams.managers

.. autodoc2-docstring:: bocoel.core.exams.managers
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`Manager <bocoel.core.exams.managers.Manager>`
     - .. autodoc2-docstring:: bocoel.core.exams.managers.Manager
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`LOGGER <bocoel.core.exams.managers.LOGGER>`
     - .. autodoc2-docstring:: bocoel.core.exams.managers.LOGGER
          :summary:

API
~~~

.. py:data:: LOGGER
   :canonical: bocoel.core.exams.managers.LOGGER
   :value: 'get_logger(...)'

   .. autodoc2-docstring:: bocoel.core.exams.managers.LOGGER

.. py:class:: Manager(root: str | pathlib.Path | None = None, skip_rerun: bool = True)
   :canonical: bocoel.core.exams.managers.Manager

   .. autodoc2-docstring:: bocoel.core.exams.managers.Manager

   .. rubric:: Initialization

   .. autodoc2-docstring:: bocoel.core.exams.managers.Manager.__init__

   .. py:attribute:: _examinator
      :canonical: bocoel.core.exams.managers.Manager._examinator
      :type: bocoel.core.exams.examinators.Examinator
      :value: None

      .. autodoc2-docstring:: bocoel.core.exams.managers.Manager._examinator

   .. py:method:: run(steps: int | None = None, *, optimizer: bocoel.core.optim.Optimizer, embedder: bocoel.corpora.Embedder, corpus: bocoel.corpora.Corpus, model: bocoel.models.GenerativeModel | bocoel.models.ClassifierModel, adaptor: bocoel.models.Adaptor) -> pandas.DataFrame
      :canonical: bocoel.core.exams.managers.Manager.run

      .. autodoc2-docstring:: bocoel.core.exams.managers.Manager.run

   .. py:method:: save(*, scores: pandas.DataFrame, optimizer: bocoel.core.optim.Optimizer, corpus: bocoel.corpora.Corpus, model: bocoel.models.GenerativeModel | bocoel.models.ClassifierModel, adaptor: bocoel.models.Adaptor, embedder: bocoel.corpora.Embedder, md5: str) -> None
      :canonical: bocoel.core.exams.managers.Manager.save

      .. autodoc2-docstring:: bocoel.core.exams.managers.Manager.save

   .. py:method:: with_cols(df: pandas.DataFrame, columns: dict[str, typing.Any]) -> pandas.DataFrame
      :canonical: bocoel.core.exams.managers.Manager.with_cols

      .. autodoc2-docstring:: bocoel.core.exams.managers.Manager.with_cols

   .. py:method:: _launch(optimizer: bocoel.core.optim.Optimizer, steps: int | None = None) -> collections.abc.Generator[collections.abc.Mapping[int, float], None, None]
      :canonical: bocoel.core.exams.managers.Manager._launch
      :staticmethod:

      .. autodoc2-docstring:: bocoel.core.exams.managers.Manager._launch

   .. py:method:: load(path: str | pathlib.Path) -> pandas.DataFrame
      :canonical: bocoel.core.exams.managers.Manager.load
      :staticmethod:

      .. autodoc2-docstring:: bocoel.core.exams.managers.Manager.load

   .. py:method:: md5(*, optimizer: bocoel.core.optim.Optimizer, embedder: bocoel.corpora.Embedder, corpus: bocoel.corpora.Corpus, model: bocoel.models.GenerativeModel | bocoel.models.ClassifierModel, adaptor: bocoel.models.Adaptor) -> str
      :canonical: bocoel.core.exams.managers.Manager.md5
      :staticmethod:

      .. autodoc2-docstring:: bocoel.core.exams.managers.Manager.md5

   .. py:method:: current() -> str
      :canonical: bocoel.core.exams.managers.Manager.current
      :staticmethod:

      .. autodoc2-docstring:: bocoel.core.exams.managers.Manager.current
