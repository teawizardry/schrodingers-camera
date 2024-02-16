# Schrödinger's Camera: First Steps Towards a Quantum-Based Privacy Preserving Camera

>*Privacy-preserving vision must overcome the dual challenge of utility and privacy. Too much anonymity renders the images useless, but too little privacy does not protect sensitive data. We propose a novel design for privacy preservation, where the imagery is stored in quantum states. In the future, this will be enabled by quantum imaging cameras, and, currently, storing very low resolution imagery in quantum states is possible. Quantum state imagery has the advantage of being both private and non-private till the point of measurement. This occurs even when images are manipulated, since every quantum action is fully reversible. We propose a control algorithm, based on double deep Q-learning, to learn how to anonymize the image before measurement. After learning, the RL weights are fixed, and new attack neural networks are trained from scratch to break the system's privacy. Although all our results are in simulation, we demonstrate, with these first steps, that it is possible to control both privacy and utility in a quantum-based manner.*

[CVPRW2023](https://openaccess.thecvf.com/content/CVPR2023W/TCV/html/Kirkland_Schrodingers_Camera_First_Steps_Towards_a_Quantum-Based_Privacy_Preserving_Camera_CVPRW_2023_paper.html_ | [arXiv](https://arxiv.org/abs/2303.07510) | [Poster](https://focus.ece.ufl.edu/wp-content/uploads/2023/06/poster.pdf) | [Presentation](https://share.descript.com/view/F4Z6N2GhvvJ)

---

**File Structure**
- `agent.py`: DDQN agent
- `helper.py`: Logging and QOL functions
- `neural.py`: Models for the agent and environment
- `QiskitHelper.py`: Functions to encode images via FRQI and decode Qiskit simulation output
- `RL.py`: Main training file
- `test.py`: Main testing file

To train, extract the two `.zip` files in `./aug_sets/` and `./resouces/` then run `python RL.py`. You can specify different arguments such as `-bi` for the burn in length (see the top of `RL.py` for a full list).

**Important Compatibility Notes**
- See `requirements.txt` for Python module requirements.
- Qiskit GPU support is limited to x84_64 Linux. See the [`qiskit-aer-gpu`](https://pypi.org/project/qiskit-aer-gpu/) page for more info.
	- To run using CPU, change `device='GPU'` to `device='CPU'` in line 127 of `quantumEnv.py`. Alternatively, use the standard `quiskit-aer` package and leave out the line entirely.

*The PyTorch [Train A Mario-Playing RL Agent](https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html) is a helpful guide to to implementing the DDQN algorithm*

---

**`QiskitHelper.py` Functions**

This class can be used separate from this project with minimal changes for 16x16 FRQI images. For implementation information, please see [this paper](https://link.springer.com/article/10.1007/s11128-023-03838-0).
```
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, Aer, transpile
from QiskitHelper import FRQIHelper
import numpy as np

frqi = FRQIHelper()
image = np.zeros((16,16))

# set up quantum
c = QuantumRegister(1, 'c')
p = QuantumRegister(8, 'p')
cl = ClassicalRegister(9)
circ = QuantumCircuit(c, p, cl)

# frqi
action = 99 # this is arbitrary, it is simply saying to keep the image clear
frqi.frqi_encoder(circ, p, c, image.flatten(), action)
circ.barrier()

# measure the image
circ.measure(c, cl[0])
circ.measure(p, cl[1:9])

# simulation
aer_sim = Aer.get_backend('aer_simulator')
aer_sim.set_options(device='GPU')
t_qc = transpile(self.circ, aer_sim)
shots = 4096
result = aer_sim.run(t_qc, shots=shots).result()
counts = result.get_counts(circ)

# generated image
meas_img = frqi.frqi_decode(shots, counts, len(image)**2, inverse_norm=True)
```
