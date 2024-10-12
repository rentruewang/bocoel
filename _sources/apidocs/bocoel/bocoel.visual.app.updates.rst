:py:mod:`bocoel.visual.app.updates`
===================================

.. py:module:: bocoel.visual.app.updates

.. autodoc2-docstring:: bocoel.visual.app.updates
   :allowtitles:

Module Contents
---------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`control_text_1 <bocoel.visual.app.updates.control_text_1>`
     - .. autodoc2-docstring:: bocoel.visual.app.updates.control_text_1
          :summary:
   * - :py:obj:`control_text_2 <bocoel.visual.app.updates.control_text_2>`
     - .. autodoc2-docstring:: bocoel.visual.app.updates.control_text_2
          :summary:
   * - :py:obj:`table <bocoel.visual.app.updates.table>`
     - .. autodoc2-docstring:: bocoel.visual.app.updates.table
          :summary:
   * - :py:obj:`two_d <bocoel.visual.app.updates.two_d>`
     - .. autodoc2-docstring:: bocoel.visual.app.updates.two_d
          :summary:
   * - :py:obj:`x_splines <bocoel.visual.app.updates.x_splines>`
     - .. autodoc2-docstring:: bocoel.visual.app.updates.x_splines
          :summary:
   * - :py:obj:`y_splines <bocoel.visual.app.updates.y_splines>`
     - .. autodoc2-docstring:: bocoel.visual.app.updates.y_splines
          :summary:
   * - :py:obj:`three_d_single <bocoel.visual.app.updates.three_d_single>`
     - .. autodoc2-docstring:: bocoel.visual.app.updates.three_d_single
          :summary:
   * - :py:obj:`three_d <bocoel.visual.app.updates.three_d>`
     - .. autodoc2-docstring:: bocoel.visual.app.updates.three_d
          :summary:

API
~~~

.. py:function:: control_text_1(slider_value: float)
   :canonical: bocoel.visual.app.updates.control_text_1

   .. autodoc2-docstring:: bocoel.visual.app.updates.control_text_1

.. py:function:: control_text_2(slider_value: float) -> plotly.graph_objects.Figure
   :canonical: bocoel.visual.app.updates.control_text_2

   .. autodoc2-docstring:: bocoel.visual.app.updates.control_text_2

.. py:function:: table(slider_value: float, df: pandas.DataFrame) -> dash.dash_table.DataTable
   :canonical: bocoel.visual.app.updates.table

   .. autodoc2-docstring:: bocoel.visual.app.updates.table

.. py:function:: two_d(slider_value: float, df: pandas.DataFrame) -> plotly.graph_objects.Figure
   :canonical: bocoel.visual.app.updates.two_d

   .. autodoc2-docstring:: bocoel.visual.app.updates.two_d

.. py:function:: x_splines(slider_value: float, df: pandas.DataFrame) -> plotly.graph_objects.Figure
   :canonical: bocoel.visual.app.updates.x_splines

   .. autodoc2-docstring:: bocoel.visual.app.updates.x_splines

.. py:function:: y_splines(slider_value: float, df: pandas.DataFrame) -> plotly.graph_objects.Figure
   :canonical: bocoel.visual.app.updates.y_splines

   .. autodoc2-docstring:: bocoel.visual.app.updates.y_splines

.. py:function:: three_d_single(slider_value: float, ci: float, dfs: list, row: int, col: int, names: list) -> plotly.graph_objects.Figure
   :canonical: bocoel.visual.app.updates.three_d_single

   .. autodoc2-docstring:: bocoel.visual.app.updates.three_d_single

.. py:function:: three_d(slider_value: float, ci: float, llm: collections.abc.Sequence[str], corpus: collections.abc.Sequence[str], layout_children: collections.abc.Sequence[str], data: collections.abc.Sequence[str])
   :canonical: bocoel.visual.app.updates.three_d

   .. autodoc2-docstring:: bocoel.visual.app.updates.three_d
