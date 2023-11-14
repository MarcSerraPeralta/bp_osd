import stim
import numpy as np

from pymatching import Matching
from bp_osd import BP_OSD


def test_bp_osd_performance():
    circuit = stim.Circuit.generated(
        code_task="surface_code:rotated_memory_z",
        distance=3,
        rounds=3,
        after_clifford_depolarization=0.01,
    )

    sampler = circuit.compile_detector_sampler()
    defects, log_flips = sampler.sample(shots=100_000, separate_observables=True)

    mwpm = Matching(circuit.detector_error_model(decompose_errors=True))
    predictions = mwpm.decode_batch(defects)
    log_prob_mwpm = np.average(predictions != log_flips)

    bp_osd_cs = BP_OSD(
        circuit.detector_error_model(), osd_method="osd_cs", osd_order=40
    )
    predictions = bp_osd_cs.decode_batch(defects)
    print(np.average(predictions != log_flips))
    log_prob_bposd = np.average(predictions != log_flips)

    assert log_prob_bposd < log_prob_mwpm

    return
