from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    from scipy import sparse
except ImportError:
    sparse = None

try:
    import h5py
except ImportError:
    h5py = None


@unittest.skipIf(sparse is None, "scipy is required for registration fixtures")
class RegistrationH5ReferenceTests(unittest.TestCase):
    @staticmethod
    def _registration_functions():
        from preprocess_functions.registration import (
            load_extract_spatial_weights,
            spatial_weights_to_caiman_A,
        )

        return load_extract_spatial_weights, spatial_weights_to_caiman_A

    @unittest.skipIf(h5py is None, "h5py is required for HDF5 reference fixtures")
    def test_loads_matlab_reference_array_with_one_sparse_target(self) -> None:
        load_extract_spatial_weights, spatial_weights_to_caiman_A = (
            self._registration_functions()
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            mat_path = Path(tmpdir) / "extract_output_unsorted.mat"
            expected = sparse.csc_matrix(
                np.array(
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, 2.0, 0.0],
                        [0.0, 0.0, 3.0],
                        [4.0, 0.0, 0.0],
                    ]
                )
            )
            with h5py.File(mat_path, "w") as h5f:
                output = h5f.create_group("output")
                refs = h5f.create_group("#refs#")
                spatial_group = refs.create_group("spatial_weights_0")
                _write_matlab_sparse_group(spatial_group, expected)

                ref_dtype = h5py.special_dtype(ref=h5py.Reference)
                ref_values = np.empty((6,), dtype=ref_dtype)
                ref_values[0] = spatial_group.ref
                for idx in range(1, 6):
                    ref_values[idx] = h5py.Reference()
                output.create_dataset("spatial_weights", data=ref_values)

            spatial_weights = load_extract_spatial_weights(mat_path)
            A, dims = spatial_weights_to_caiman_A(
                spatial_weights,
                labels=np.array([1, 0, 1]),
                template_shape=(2, 2),
            )

            self.assertEqual(dims, (2, 2))
            np.testing.assert_allclose(A.toarray(), expected.toarray())

    @unittest.skipIf(h5py is None, "h5py is required for HDF5 reference fixtures")
    def test_concatenates_multiple_referenced_sparse_targets(self) -> None:
        load_extract_spatial_weights, spatial_weights_to_caiman_A = (
            self._registration_functions()
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            mat_path = Path(tmpdir) / "extract_output_unsorted.mat"
            first = sparse.csc_matrix(
                np.array(
                    [
                        [1.0, 0.0],
                        [0.0, 2.0],
                        [0.0, 0.0],
                        [4.0, 0.0],
                    ]
                )
            )
            second = sparse.csc_matrix(
                np.array(
                    [
                        [0.0],
                        [0.0],
                        [3.0],
                        [0.0],
                    ]
                )
            )
            expected = sparse.hstack([first, second], format="csc")

            with h5py.File(mat_path, "w") as h5f:
                output = h5f.create_group("output")
                refs = h5f.create_group("#refs#")
                first_group = refs.create_group("spatial_weights_0")
                second_group = refs.create_group("spatial_weights_1")
                _write_matlab_sparse_group(first_group, first)
                _write_matlab_sparse_group(second_group, second)

                ref_dtype = h5py.special_dtype(ref=h5py.Reference)
                ref_values = np.empty((2,), dtype=ref_dtype)
                ref_values[0] = first_group.ref
                ref_values[1] = second_group.ref
                output.create_dataset("spatial_weights", data=ref_values)

            spatial_weights = load_extract_spatial_weights(mat_path)
            A, dims = spatial_weights_to_caiman_A(
                spatial_weights,
                labels=np.array([1, 1, 0]),
                template_shape=(2, 2),
            )

            self.assertEqual(dims, (2, 2))
            np.testing.assert_allclose(A.toarray(), expected.toarray())

    @unittest.skipIf(h5py is None, "h5py is required for HDF5 ndSparse fixtures")
    def test_loads_matlab_nd_sparse_placeholder_from_refs_candidate(self) -> None:
        load_extract_spatial_weights, spatial_weights_to_caiman_A = (
            self._registration_functions()
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            mat_path = Path(tmpdir) / "extract_output_unsorted.mat"
            expected = sparse.csc_matrix(
                np.array(
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, 2.0, 0.0],
                        [0.0, 0.0, 3.0],
                        [4.0, 0.0, 0.0],
                    ]
                )
            )
            partition = sparse.csc_matrix(
                np.array(
                    [
                        [5.0],
                        [0.0],
                        [0.0],
                        [0.0],
                    ]
                )
            )
            with h5py.File(mat_path, "w") as h5f:
                output = h5f.create_group("output")
                refs = h5f.create_group("#refs#")
                full_group = refs.create_group("Wo")
                partition_group = refs.create_group("b")
                _write_matlab_sparse_group(full_group, expected)
                _write_matlab_sparse_group(partition_group, partition)

                placeholder = output.create_dataset(
                    "spatial_weights",
                    data=np.array([[3707764736, 2, 1, 1, 1, 1]], dtype=np.uint32),
                )
                placeholder.attrs["H5PATH"] = "/output"
                placeholder.attrs["MATLAB_class"] = "ndSparse"
                placeholder.attrs["MATLAB_object_decode"] = np.int32(3)

            spatial_weights = load_extract_spatial_weights(
                mat_path,
                label_count=3,
                template_shape=(2, 2),
            )
            A, dims = spatial_weights_to_caiman_A(
                spatial_weights,
                labels=np.array([1, 0, 1]),
                template_shape=(2, 2),
            )

            self.assertEqual(dims, (2, 2))
            np.testing.assert_allclose(A.toarray(), expected.toarray())

    def test_concatenates_sparse_sequence_for_partitioned_spatial_weights(self) -> None:
        _, spatial_weights_to_caiman_A = self._registration_functions()
        first = sparse.csc_matrix(
            np.array(
                [
                    [1.0, 0.0],
                    [0.0, 2.0],
                    [0.0, 0.0],
                    [4.0, 0.0],
                ]
            )
        )
        second = sparse.csc_matrix(
            np.array(
                [
                    [0.0],
                    [0.0],
                    [3.0],
                    [0.0],
                ]
            )
        )
        expected = sparse.hstack([first, second], format="csc")

        A, dims = spatial_weights_to_caiman_A(
            [first, second],
            labels=np.array([1, 1, 0]),
            template_shape=(2, 2),
        )

        self.assertEqual(dims, (2, 2))
        np.testing.assert_allclose(A.toarray(), expected.toarray())


def _write_matlab_sparse_group(group, matrix: sparse.csc_matrix) -> None:
    matrix = matrix.tocsc()
    group.create_dataset("data", data=matrix.data)
    group.create_dataset("ir", data=matrix.indices.astype(np.uint64))
    group.create_dataset("jc", data=matrix.indptr.astype(np.uint64))
    group.attrs["MATLAB_sparse"] = np.array([matrix.shape[0]], dtype=np.uint64)


if __name__ == "__main__":
    unittest.main()
