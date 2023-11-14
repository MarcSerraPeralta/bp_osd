# bp_osd: Belief propagation with ordered statistics decoding post-processing

This is a wrapper for the `ldpc.bposd_decoder` to work with `stim`. 

Example of usage:

```
import stim
import numpy as np

from pymatching import Matching
from bp_osd import BP_OSD

circuit = stim.Circuit.generated(code_task="surface_code:rotated_memory_z", 
                                 distance=3, 
                                 rounds=5, 
                                 after_clifford_depolarization=0.01)


sampler = circuit.compile_detector_sampler()
defects, log_flips = sampler.sample(shots=1_000_000, separate_observables=True)

mwpm = Matching(circuit.detector_error_model(decompose_errors=True))
predictions = mwpm.decode_batch(defects)
print(np.average(predictions != log_flips)) 
# logical error probability ~ 3.45%

bp_osd_cs = BP_OSD(circuit.detector_error_model(), osd_method="osd_cs", osd_order=40)
predictions = bp_osd_cs.decode_batch(defects)
print(np.average(predictions != log_flips)) 
# logical error probability ~ 3.24%
```
