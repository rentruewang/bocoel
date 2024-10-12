:py:mod:`bocoel.core.optim.ax.acquisition.supported`
====================================================

.. py:module:: bocoel.core.optim.ax.acquisition.supported

.. autodoc2-docstring:: bocoel.core.optim.ax.acquisition.supported
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`AcquisitionFunc <bocoel.core.optim.ax.acquisition.supported.AcquisitionFunc>`
     -

API
~~~

.. py:class:: AcquisitionFunc()
   :canonical: bocoel.core.optim.ax.acquisition.supported.AcquisitionFunc

   Bases: :py:obj:`bocoel.common.StrEnum`

   .. py:attribute:: ENTROPY
      :canonical: bocoel.core.optim.ax.acquisition.supported.AcquisitionFunc.ENTROPY
      :value: 'ENTROPY'

      .. autodoc2-docstring:: bocoel.core.optim.ax.acquisition.supported.AcquisitionFunc.ENTROPY

   .. py:attribute:: MES
      :canonical: bocoel.core.optim.ax.acquisition.supported.AcquisitionFunc.MES
      :value: 'MAX_VALUE_ENTROPY'

      .. autodoc2-docstring:: bocoel.core.optim.ax.acquisition.supported.AcquisitionFunc.MES

   .. py:attribute:: UCB
      :canonical: bocoel.core.optim.ax.acquisition.supported.AcquisitionFunc.UCB
      :value: 'UPPER_CONFIDENCE_BOUND'

      .. autodoc2-docstring:: bocoel.core.optim.ax.acquisition.supported.AcquisitionFunc.UCB

   .. py:attribute:: QUCB
      :canonical: bocoel.core.optim.ax.acquisition.supported.AcquisitionFunc.QUCB
      :value: 'QUASI_UPPER_CONFIDENCE_BOUND'

      .. autodoc2-docstring:: bocoel.core.optim.ax.acquisition.supported.AcquisitionFunc.QUCB

   .. py:attribute:: EI
      :canonical: bocoel.core.optim.ax.acquisition.supported.AcquisitionFunc.EI
      :value: 'EXPECTED_IMPROVEMENT'

      .. autodoc2-docstring:: bocoel.core.optim.ax.acquisition.supported.AcquisitionFunc.EI

   .. py:attribute:: QEI
      :canonical: bocoel.core.optim.ax.acquisition.supported.AcquisitionFunc.QEI
      :value: 'QUASI_EXPECTED_IMPROVEMENT'

      .. autodoc2-docstring:: bocoel.core.optim.ax.acquisition.supported.AcquisitionFunc.QEI

   .. py:attribute:: AUTO
      :canonical: bocoel.core.optim.ax.acquisition.supported.AcquisitionFunc.AUTO
      :value: 'AUTO'

      .. autodoc2-docstring:: bocoel.core.optim.ax.acquisition.supported.AcquisitionFunc.AUTO

   .. py:property:: botorch_acqf_class
      :canonical: bocoel.core.optim.ax.acquisition.supported.AcquisitionFunc.botorch_acqf_class
      :type: type[botorch.acquisition.AcquisitionFunction] | None

      .. autodoc2-docstring:: bocoel.core.optim.ax.acquisition.supported.AcquisitionFunc.botorch_acqf_class
