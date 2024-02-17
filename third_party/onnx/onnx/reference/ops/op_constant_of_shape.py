# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy as np

from onnx.reference.op_run import OpRun


class ConstantOfShape(OpRun):
    def __init__(self, onnx_node, run_params):  # type: ignore
        OpRun.__init__(self, onnx_node, run_params)
        self.cst = (
            self.value[0] if isinstance(self.value, np.ndarray) else self.value  # type: ignore
        )
        if isinstance(self.cst, int):
            self.cst = np.int64(self.cst)
        elif isinstance(self.cst, float):
            self.cst = np.float64(self.cst)
        if not isinstance(
            self.cst,
            (
                np.float32,
                np.float64,
                np.int64,
                np.int32,
                np.bool_,
                np.float16,
            ),
        ):
            raise TypeError(f"cst must be a real not {type(self.cst)}")

    def _run(self, data, value=None):  # type: ignore
        try:
            res = np.full(tuple(data), self.cst)  # type: ignore
        except TypeError as e:
            raise RuntimeError(
                f"Unable to create a constant of shape {data!r} with value {self.cst!r} "  # type: ignore
                f"(raw value={value!r})."  # type: ignore
            ) from e
        return (res,)
