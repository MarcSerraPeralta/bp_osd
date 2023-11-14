from typing import List, FrozenSet, Dict, Tuple, Union
from dataclasses import dataclass
from tqdm import tqdm

from ldpc import bposd_decoder
from scipy.sparse import csc_matrix
import numpy as np

import stim


def iter_set_xor(set_list: List[List[int]]) -> FrozenSet[int]:
    out = set()
    for x in set_list:
        s = set(x)
        out = (out - s) | (s - out)
    return frozenset(out)


def dict_to_csc_matrix(
    elements_dict: Dict[int, FrozenSet[int]], shape: Tuple[int, int]
) -> csc_matrix:
    """
    Constructs a `scipy.sparse.csc_matrix` check matrix from a dictionary `elements_dict` giving the indices of nonzero
    rows in each column.

    Parameters
    ----------
    elements_dict
        A dictionary giving the indices of nonzero rows in each column. `elements_dict[i]` is a frozenset of ints
        giving the indices of nonzero rows in column `i`.
    shape
        The dimensions of the matrix to be returned

    Returns
    -------
    scipy.sparse.csc_matrix
        The `scipy.sparse.csc_matrix` check matrix defined by `elements_dict` and `shape`
    """
    nnz = sum(len(v) for k, v in elements_dict.items())
    data = np.ones(nnz, dtype=np.uint8)
    row_ind = np.zeros(nnz, dtype=np.int64)
    col_ind = np.zeros(nnz, dtype=np.int64)
    i = 0
    for col, v in elements_dict.items():
        for row in v:
            row_ind[i] = row
            col_ind[i] = col
            i += 1
    return csc_matrix((data, (row_ind, col_ind)), shape=shape)


@dataclass
class DemMatrices:
    check_matrix: csc_matrix
    observables_matrix: csc_matrix
    priors: np.ndarray


def detector_error_model_to_check_matrices(
    dem: stim.DetectorErrorModel,
) -> DemMatrices:
    hyperedge_ids: Dict[FrozenSet[int], int] = {}
    hyperedge_obs_map: Dict[int, FrozenSet[int]] = {}
    priors_dict: Dict[int, float] = {}

    def handle_error(
        prob: float, detectors: List[List[int]], observables: List[List[int]]
    ) -> None:
        hyperedge_dets = iter_set_xor(detectors)
        hyperedge_obs = iter_set_xor(observables)

        # to avid multiple (hyper)edges being repeated inside the DEM file
        if hyperedge_dets not in hyperedge_ids:
            hyperedge_ids[hyperedge_dets] = len(hyperedge_ids)  # create new id
            priors_dict[hyperedge_ids[hyperedge_dets]] = 0.0

        hid = hyperedge_ids[hyperedge_dets]
        hyperedge_obs_map[hid] = hyperedge_obs
        priors_dict[hid] = priors_dict[hid] * (1 - prob) + prob * (1 - priors_dict[hid])
        return

    for instruction in dem.flattened():
        if instruction.type == "error":
            dets: List[List[int]] = [[]]
            frames: List[List[int]] = [[]]
            t: stim.DemTarget
            p = instruction.args_copy()[0]
            for t in instruction.targets_copy():
                if t.is_relative_detector_id():
                    dets[-1].append(t.val)
                elif t.is_logical_observable_id():
                    frames[-1].append(t.val)
                elif t.is_separator():
                    dets.append([])
                    frames.append([])
            handle_error(p, dets, frames)
        elif instruction.type == "detector":
            pass
        elif instruction.type == "logical_observable":
            pass
        else:
            raise NotImplementedError()

    check_matrix = dict_to_csc_matrix(
        {v: k for k, v in hyperedge_ids.items()},
        shape=(dem.num_detectors, len(hyperedge_ids)),
    )
    observables_matrix = dict_to_csc_matrix(
        hyperedge_obs_map, shape=(dem.num_observables, len(hyperedge_ids))
    )
    priors = np.zeros(len(hyperedge_ids))
    for i, p in priors_dict.items():
        priors[i] = p

    return DemMatrices(
        check_matrix=check_matrix,
        observables_matrix=observables_matrix,
        priors=priors,
    )


class BP_OSD:
    def __init__(
        self,
        model: Union[stim.Circuit, stim.DetectorErrorModel],
        max_bp_iters: int = 20,
        bp_method: str = "minimum_sum",
        osd_method: str = "osd0",
        **kwargs
    ):
        """
        Construct a BP_OSD object from a `stim.Circuit` or `stim.DetectorErrorModel`

        Parameters
        ----------
        model
            A stim.Circuit or a stim.DetectorErrorModel. If a stim.Circuit is provided, it will be converted
            into a stim.DetectorErrorModel using `stim.Circuit.detector_error_model()`.
        max_bp_iters
            The maximum number of interations of belief-propagation to use. Passed to
            `ldpc.bposd_decoder` as the `max_iter` argument. Default 20
        bp_method
            The method of belief-propagation to use. Passed to
            `ldpc.bp_decoder` as the `bp_method` argument. Options include "product_sum",
             "minimum_sum", "product_sum_log", and "minimum_sum_log" (see https://github.com/quantumgizmos/ldpc
             for details). Default is "minimum_sum" as in https://arxiv.org/pdf/2308.07915.pdf.
        osd_method
            The method of ordered statistics decoding to use. Paseed to
            `ldpc.bposd_decoder` as the `osd_method` argument. Options include "osd0",
             "osd_cs", and "osd_e" (see https://github.com/quantumgizmos/ldpc
             for details). Default is "osd0".
        kwargs
            Additional keyword arguments are passed to `ldpc.bposd_decoder`
        """
        if isinstance(model, stim.Circuit):
            model = model.detector_error_model()
        self._initialise_from_detector_error_model(
            model=model,
            max_bp_iters=max_bp_iters,
            bp_method=bp_method,
            osd_method=osd_method,
            **kwargs,
        )

    def _initialise_from_detector_error_model(
        self,
        model: stim.DetectorErrorModel,
        *,
        max_bp_iters: int = 20,
        bp_method: str = "minimum_sum",
        osd_method: str = "osd0",
        **kwargs
    ):
        self._model = model
        self._matrices = detector_error_model_to_check_matrices(self._model)
        self._bp_osd = bposd_decoder(
            self._matrices.check_matrix,
            max_iter=max_bp_iters,
            bp_method=bp_method,
            channel_probs=self._matrices.priors,
            osd_method=osd_method,
            **kwargs,
        )

    @classmethod
    def from_detector_error_model(
        cls,
        model: stim.DetectorErrorModel,
        *,
        max_bp_iters: int = 20,
        bp_method: str = "minimum_sum",
        osd_method: str = "osd0",
        **kwargs
    ) -> "BP_OSD":
        """
        Construct a BP_OSD object from a `stim.DetectorErrorModel`

        Parameters
        ----------
        model : stim.DetectorErrorModel
            A `stim.DetectorErrorModel`. It does not matter if the hyperedges are decomposed.
        max_bp_iters : int
            The maximum number of interations of belief-propagation to use. Passed to
            `ldpc.bp_decoder` as the `max_iter` argument. Default 20
        bp_method
            The method of belief-propagation to use. Passed to
            `ldpc.bp_decoder` as the `bp_method` argument. Options include "product_sum",
             "minimum_sum", "product_sum_log", and "minimum_sum_log" (see https://github.com/quantumgizmos/ldpc
             for details). Default is "minimum_sum" as in https://arxiv.org/pdf/2308.07915.pdf.
        osd_method
            The method of ordered statistics decoding to use. Paseed to
            `ldpc.bposd_decoder` as the `osd_method` argument. Options include "osd0",
             "osd_cs", and "osd_e" (see https://github.com/quantumgizmos/ldpc
             for details). Default is "osd0".
        kwargs
            Additional keyword arguments are passed to `ldpc.bposd_decoder`


        Returns
        -------
        BP_OSD
            The BP_OSD object for decoding using `model`
        """
        bm = cls.__new__(cls)
        bm._initialise_from_detector_error_model(
            model=model,
            max_bp_iters=max_bp_iters,
            bp_method=bp_method,
            osd_method=osd_method,
            **kwargs,
        )
        return bm

    @classmethod
    def from_stim_circuit(
        cls,
        circuit: stim.Circuit,
        *,
        max_bp_iters: int = 20,
        bp_method: str = "minimum_sum",
        osd_method: str = "osd0",
        **kwargs
    ) -> "BP_OSD":
        """
        Construct a BP_OSD object from a `stim.Circuit`

        Parameters
        ----------
        circuit : stim.Circuit
            A stim.Circuit. The circuit will be converted into a stim.DetectorErrorModel using
            `stim.Circuit.detector_error_model()`.
        max_bp_iters : int
            The maximum number of interations of belief-propagation to use. Passed to
            `ldpc.bp_decoder` as the `max_iter` argument. Default 20
        bp_method
            The method of belief-propagation to use. Passed to
            `ldpc.bp_decoder` as the `bp_method` argument. Options include "product_sum",
             "minimum_sum", "product_sum_log", and "minimum_sum_log" (see https://github.com/quantumgizmos/ldpc
             for details). Default is "minimum_sum" as in https://arxiv.org/pdf/2308.07915.pdf.
        osd_method
            The method of ordered statistics decoding to use. Paseed to
            `ldpc.bposd_decoder` as the `osd_method` argument. Options include "osd0",
             "osd_cs", and "osd_e" (see https://github.com/quantumgizmos/ldpc
             for details). Default is "osd0".
        kwargs
            Additional keyword arguments are passed to `ldpc.bposd_decoder`


        Returns
        -------
        BP_OSD
            The BP_OSD object for decoding using `model`
        """
        bm = cls.__new__(cls)
        model = circuit.detector_error_model()
        bm._initialise_from_detector_error_model(
            model=model,
            max_bp_iters=max_bp_iters,
            bp_method=bp_method,
            osd_method=osd_method,
            **kwargs,
        )
        return bm

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        """
        Decode the syndrome and return a prediction of which observables were flipped

        Parameters
        ----------
        syndrome : np.ndarray
            A single shot of syndrome data. This should be a binary array with a length equal to the
            number of detectors in the `stim.Circuit` or `stim.DetectorErrorModel`. E.g. the syndrome might be
            one row of shot data sampled from a `stim.CompiledDetectorSampler`.

        Returns
        -------
        np.ndarray
            A binary numpy array `predictions` which predicts which observables were flipped.
            Its length is equal to the number of observables in the `stim.Circuit` or `stim.DetectorErrorModel`.
            `predictions[i]` is 1 if the decoder predicts observable `i` was flipped and 0 otherwise.
        """
        error_mechanisms = self._bp_osd.decode(syndrome)
        logical_errors = (self._matrices.observables_matrix @ error_mechanisms) % 2
        return logical_errors

    def decode_to_faults_array(self, syndrome: np.ndarray) -> np.ndarray:
        """
        Decode the syndrome and return a prediction of which faults were triggered

        Parameters
        ----------
        syndrome : np.ndarray
            A single shot of syndrome data. This should be a binary array with a length equal to the
            number of detectors in the `stim.Circuit` or `stim.DetectorErrorModel`. E.g. the syndrome might be
            one row of shot data sampled from a `stim.CompiledDetectorSampler`.

        Returns
        -------
        np.ndarray
            A binary numpy array `predictions` which predicts which faults were triggered.
            Its length is equal to the number of observables in the `stim.Circuit` or `stim.DetectorErrorModel`.
            `predictions[i]` is 1 if the decoder predicts fault `i` was triggered and 0 otherwise.
        """
        return self._bp_osd.decode(syndrome)

    def decode_batch(self, shots: np.ndarray, verbose=True) -> np.ndarray:
        """
        Decode a batch of shots of syndrome data. This is just a helper method, equivalent to iterating over each
        shot and calling `BP_OSD.decode` on it.

        Parameters
        ----------
        shots : np.ndarray
            A binary numpy array of dtype `np.uint8` or `bool` with shape `(num_shots, num_detectors)`, where
            here `num_shots` is the number of shots and `num_detectors` is the number of detectors in the `stim.Circuit` or `stim.DetectorErrorModel`.

        Returns
        -------
        np.ndarray
            A 2D numpy array `predictions` of dtype bool, where `predictions[i, :]` is the output of
            `self.decode(shots[i, :])`.
        """
        predictions = np.zeros(
            (shots.shape[0], self._matrices.observables_matrix.shape[0]), dtype=bool
        )
        if verbose:
            for i in tqdm(range(shots.shape[0])):
                predictions[i, :] = self.decode(shots[i, :])
        else:
            for i in range(shots.shape[0]):
                predictions[i, :] = self.decode(shots[i, :])
        return predictions
