:py:mod:`bocoel.core.optim.ax.params`
=====================================

.. py:module:: bocoel.core.optim.ax.params

.. autodoc2-docstring:: bocoel.core.optim.ax.params
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`AxServiceParameter <bocoel.core.optim.ax.params.AxServiceParameter>`
     - .. autodoc2-docstring:: bocoel.core.optim.ax.params.AxServiceParameter
          :summary:

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`configs <bocoel.core.optim.ax.params.configs>`
     - .. autodoc2-docstring:: bocoel.core.optim.ax.params.configs
          :summary:
   * - :py:obj:`name_dict <bocoel.core.optim.ax.params.name_dict>`
     - .. autodoc2-docstring:: bocoel.core.optim.ax.params.name_dict
          :summary:
   * - :py:obj:`name_list <bocoel.core.optim.ax.params.name_list>`
     - .. autodoc2-docstring:: bocoel.core.optim.ax.params.name_list
          :summary:
   * - :py:obj:`name <bocoel.core.optim.ax.params.name>`
     - .. autodoc2-docstring:: bocoel.core.optim.ax.params.name
          :summary:

API
~~~

.. py:class:: AxServiceParameter()
   :canonical: bocoel.core.optim.ax.params.AxServiceParameter

   Bases: :py:obj:`typing.TypedDict`

   .. autodoc2-docstring:: bocoel.core.optim.ax.params.AxServiceParameter

   .. rubric:: Initialization

   .. autodoc2-docstring:: bocoel.core.optim.ax.params.AxServiceParameter.__init__

   .. py:attribute:: name
      :canonical: bocoel.core.optim.ax.params.AxServiceParameter.name
      :type: str
      :value: None

      .. autodoc2-docstring:: bocoel.core.optim.ax.params.AxServiceParameter.name

   .. py:attribute:: type
      :canonical: bocoel.core.optim.ax.params.AxServiceParameter.type
      :type: str
      :value: None

      .. autodoc2-docstring:: bocoel.core.optim.ax.params.AxServiceParameter.type

   .. py:attribute:: bounds
      :canonical: bocoel.core.optim.ax.params.AxServiceParameter.bounds
      :type: tuple[float, float]
      :value: None

      .. autodoc2-docstring:: bocoel.core.optim.ax.params.AxServiceParameter.bounds

   .. py:attribute:: value_type
      :canonical: bocoel.core.optim.ax.params.AxServiceParameter.value_type
      :type: typing_extensions.NotRequired[str]
      :value: None

      .. autodoc2-docstring:: bocoel.core.optim.ax.params.AxServiceParameter.value_type

   .. py:attribute:: log_scale
      :canonical: bocoel.core.optim.ax.params.AxServiceParameter.log_scale
      :type: typing_extensions.NotRequired[bool]
      :value: None

      .. autodoc2-docstring:: bocoel.core.optim.ax.params.AxServiceParameter.log_scale

.. py:function:: configs(boundary: bocoel.corpora.Boundary) -> list[dict[str, typing.Any]]
   :canonical: bocoel.core.optim.ax.params.configs

   .. autodoc2-docstring:: bocoel.core.optim.ax.params.configs

.. py:function:: name_dict(boundary: bocoel.corpora.Boundary, i: int) -> dict[str, typing.Any]
   :canonical: bocoel.core.optim.ax.params.name_dict

   .. autodoc2-docstring:: bocoel.core.optim.ax.params.name_dict

.. py:function:: name_list(total: int) -> list[str]
   :canonical: bocoel.core.optim.ax.params.name_list

   .. autodoc2-docstring:: bocoel.core.optim.ax.params.name_list

.. py:function:: name(number: int) -> str
   :canonical: bocoel.core.optim.ax.params.name

   .. autodoc2-docstring:: bocoel.core.optim.ax.params.name
