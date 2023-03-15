from qiskit import QuantumRegister, QuantumCircuit, Aer, IBMQ
from qiskit.tools.jupyter import *
from qiskit.visualization import plot_histogram
import numpy as np
from math import pi
import torch
from torch.nn.functional import normalize

class FRQIHelper:
    def __init__(self):
        # this variable
        self.state = None
        self.rng = np.random.default_rng()

        # get actions lut
        self.actions_lut = np.load('./resources/actions_lut.npy')

        # Build the RCCX sub-circuit
        rccx_q = QuantumRegister(3)
        rccx_circ = QuantumCircuit(rccx_q, name='RCCX')
        rccx_circ.h(rccx_q[0])
        rccx_circ.t(rccx_q[0])
        rccx_circ.cx(rccx_q[2], rccx_q[0])
        rccx_circ.tdg(rccx_q[0])
        rccx_circ.cx(rccx_q[1], rccx_q[0])
        rccx_circ.t(rccx_q[0])
        rccx_circ.cx(rccx_q[2], rccx_q[0])
        rccx_circ.tdg(rccx_q[0])
        rccx_circ.h(rccx_q[0])

        # Convert to a gate and stick it into an arbitrary place in the bigger circuit
        self.rccx_inst = rccx_circ.to_instruction()

        # Build the RC3X sub-circuit
        rc3x_q = QuantumRegister(4)
        rc3x_circ = QuantumCircuit(rc3x_q, name='RC3X')
        rc3x_circ.h(rc3x_q[0])
        rc3x_circ.t(rc3x_q[0])
        rc3x_circ.cx(rc3x_q[3], rc3x_q[0])
        rc3x_circ.tdg(rc3x_q[0])
        rc3x_circ.h(rc3x_q[0])
        rc3x_circ.cx(rc3x_q[2], rc3x_q[0])
        rc3x_circ.t(rc3x_q[0])
        rc3x_circ.cx(rc3x_q[1], rc3x_q[0])
        rc3x_circ.tdg(rc3x_q[0])
        rc3x_circ.cx(rc3x_q[2], rc3x_q[0])
        rc3x_circ.t(rc3x_q[0])
        rc3x_circ.cx(rc3x_q[1], rc3x_q[0])
        rc3x_circ.tdg(rc3x_q[0])
        rc3x_circ.h(rc3x_q[0])
        rc3x_circ.t(rc3x_q[0])
        rc3x_circ.cx(rc3x_q[3], rc3x_q[0])
        rc3x_circ.tdg(rc3x_q[0])
        rc3x_circ.h(rc3x_q[0])

        # Convert to a gate and stick it into an arbitrary place in the bigger circuit
        self.rc3x_inst = rc3x_circ.to_instruction()

    def rccx(self, circ, t, c1, c2):
        '''
        :param t: color qubit / q0, c1: q1, c2: q2
        '''
        circ.append(self.rccx_inst, [t, c1, c2])

    def rc3x(self, circ, t, c1, c2, c3):
        '''
        :param t: color qubit / q0, c1: q1, c2: q2, c3: q3
        '''
        circ.append(self.rc3x_inst, [t, c1, c2, c3])

    def mary9(self, circ, theta, t, c1, c2, c3, c4, c5, c6, c7, c8):
        '''
        mary9 gate
        '''
        circ.h(t)
        circ.t(t)
        self.rccx(circ, t, c1, c2)
        circ.tdg(t)
        circ.h(t)
        self.rc3x(circ, t, c3, c4, c5)
        circ.rz(2*theta/4, t)
        self.rc3x(circ, t, c6, c7, c8)
        circ.rz(-2*theta/4, t)
        self.rc3x(circ, t, c3, c4, c5)
        circ.rz(2*theta/4, t)
        self.rc3x(circ, t, c6, c7, c8)
        circ.rz(-2*theta/4, t)
        circ.h(t)
        circ.t(t)
        self.rccx(circ, t, c1, c2)
        circ.tdg(t)
        circ.h(t)
    
    def qreg_incrementor(self, state, circ, p):
        '''
        :param state: current state as int
        '''
        new_state = '{0:08b}'.format(int(state)+1)
        state = '{0:08b}'.format(int(state)) # binary version str

        n = len(state)  
        ind = np.array([]) 
        for i in range(n):  
            if state[i] != new_state[i]:  # get list of different values
                ind = np.append(ind, int(i))

        if len(ind) > 0:
            for i in range(len(ind)):
                circ.x(p[7-int(ind[i])])

            return
        else:
            return 

    def frqi_encoder(self, circ, p, c, image, redact_function_number):
        '''
        :param circ: quantum circuit
        :param p: position register
        :param c: color qubit
        :param image: normalized 16x16 image
        :param n: percentage of image to redact randomly
        :param redact_function_number: choose built in redactions - 
            0 - thin vertical lines
            1 - vertical lines
            2 - thick vertical lines
            3 - left half redacted
            4 - thin horizontal lines
            5 - horizontal lines
            6 - thick horizontal lines
            7 - bottom half
            8 - 27 random redaction
            else - no redaction
        '''

        # get action array
        if redact_function_number != -1:
            redact_function_array = self.actions_lut[redact_function_number]
        else:
            redact_function_array = [-1, -1, -1, -1]
        print(redact_function_array)

        # selct redacted pixels
        redaction = np.zeros(image.size)
        self.num_of_actions = 0
        for action in redact_function_array:
            if action != -1:
                self.num_of_actions += 1
                
            if action > 7 and action < 28:
                ind = self.rng.choice(redaction.size, int(((action-8)*0.05)*redaction.size), replace=False)
                redaction[ind] = 1

        # hadamard gates
        for qubit in range(p.size):
            circ.h(p[qubit])

        circ.barrier()

        # pixel encoding
        for i in range(image.size):
            if redaction[i] == 0:
                if image[i] != 0:
                    # get angle
                    theta = (pi/2)*image[i]

                    # apply MARY gate
                    self.mary9(circ, theta, c[0], p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7])
            else:
                theta = (pi/2)

                # apply MARY gate
                self.mary9(circ, theta, c[0], p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7])

            circ.barrier()

            # increment position
            self.qreg_incrementor(i, circ, p)

            circ.barrier()
            
        self.state = image.size-1

        for action in redact_function_array:
            if action == 0:
                circ.crx(pi/2,p[0],c)
            elif action == 1:
                circ.crx(pi/2,p[1],c)
            elif action == 2:
                circ.crx(pi/2,p[2],c)
            elif action == 3:
                circ.crx(pi/2,p[3],c)
            elif action == 4:
                circ.crx(pi/2,p[4],c)
            elif action == 5:
                circ.crx(pi/2,p[5],c)
            elif action == 6:
                circ.crx(pi/2,p[6],c)
            elif action == 7:
                circ.crx(pi/2,p[7],c)
            else:
                pass

        return
    
    def frqi_decode(self, shots, counts, image_size, inverse_norm=None):
        meas_img = np.array([])
        # decode
        for i in range(image_size):
            # print(format(i, '08b')+'1')
            try:
                    meas_img = np.append(meas_img,[np.sqrt(counts[format(i, '08b')+'1']/shots)])
                    # ok so the last qubit is color, the ones before that are position. It's basically backwards
                    # p7 p6 p5 p4 p3 p2 p1 p0 c0
            except KeyError:
                    meas_img = np.append(meas_img,[0.0])
        
        if inverse_norm:
            meas_img *= 16.0 * 255.0
            meas_img = meas_img.astype('int')
            img = meas_img[int(image_size/2):][::-1]
            test = meas_img[:int(image_size/2)][::-1]
            img = np.append(test, img)
            meas_img = img.reshape((16,16))
            return meas_img
        else:
            meas_img = meas_img.reshape((16,16))
            return meas_img