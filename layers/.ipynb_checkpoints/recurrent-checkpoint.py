# -*- coding: utf-8 -*-
import numpy as np

from layers.core import Layer
import activations
import initializations
from initializations import _one
from initializations import _zero


class Recurrent(Layer):
    """A recurrent neural network (RNN) is a class of artificial neural 
    network where connections between units form a directed cycle. 
    This creates an internal state of the network which allows it to 
    exhibit dynamic temporal behavior. Unlike feedforward neural networks, 
    RNNs can use their internal memory to process arbitrary sequences of 
    inputs. This makes them applicable to tasks such as unsegmented 
    connected handwriting recognition[1]_ or speech recognition.[2]_
    
    Parameters
    ----------
    n_out : int
        hidden number
    n_in : int or None
        input dimension
    nb_batch : int or None
        batch size
    nb_seq : int or None
        sequent length
    init : npdl.intializations.Initliazer
        init function
    inner_init : npdl.intializations.Initliazer
        inner init function, between hidden to hidden
    activation : npdl.activations.Activation
        activation function
    return_sequence : bool
        return total sequence or not.
    
    References
    ----------
    .. [1] A. Graves, M. Liwicki, S. Fernandez, R. Bertolami, H. Bunke, 
            J. Schmidhuber. A Novel Connectionist System for Improved 
            Unconstrained Handwriting Recognition. IEEE Transactions on 
            Pattern Analysis and Machine Intelligence, vol. 31, no. 5, 2009.
    .. [2] H. Sak and A. W. Senior and F. Beaufays. Long short-term memory 
            recurrent neural network architectures for large scale acoustic 
            modeling. Proc. Interspeech, pp338-342, Singapore, Sept. 2010
       
    """

    def __init__(self, n_out, n_in=None, nb_batch=None, nb_seq=None,
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='tanh', return_sequence=False):
        self.n_out = n_out
        self.n_in = n_in
        self.nb_batch = nb_batch
        self.nb_seq = nb_seq
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation_cls = activations.get(activation).__class__
        self.activation = activations.get(activation)
        self.return_sequence = return_sequence

        self.out_shape = None
        self.last_input = None
        self.last_output = None

    def connect_to(self, prev_layer=None):
        if prev_layer is not None:
            assert len(prev_layer.out_shape) == 3
            self.n_in = prev_layer.out_shape[-1]
            self.nb_batch = prev_layer.out_shape[0] or self.nb_batch
            self.nb_seq = prev_layer.out_shape[1] or self.nb_seq

        else:
            # print(self.n_in)
            assert self.n_in is not None

        if self.return_sequence:
            self.out_shape = (self.nb_batch, self.nb_seq, self.n_out)
        else:
            self.out_shape = (self.nb_batch, self.n_out)





class BatchLSTM(Recurrent):
    """Batch LSTM, support training, but not support mask.

    Parameters
    ----------
    gate_activation : npdl.activations.Activation
        Gate activation.
    need_grad ： bool
        If `True`, will calculate gradients.
    forget_bias_num : int
        integer.

    References
    ----------
    .. [1] Sepp Hochreiter; Jürgen Schmidhuber (1997). "Long short-term 
          memory". Neural Computation. 9 (8): 1735–1780. doi:10.1162/ne
          co.1997.9.8.1735. PMID 9377276.
    .. [2] Felix A. Gers; Jürgen Schmidhuber; Fred Cummins (2000). "Learning 
          to Forget: Continual Prediction with LSTM". Neural Computation. 12 
          (10): 2451–2471. doi:10.1162/089976600300015015.
    """

    def __init__(self, gate_activation='sigmoid', need_grad=True,
                 forget_bias_num=1, **kwargs):
        super(BatchLSTM, self).__init__(**kwargs)

        self.gate_activation_cls = activations.get(gate_activation).__class__
        self.gate_activation = activations.get(gate_activation)
        self.need_grad = need_grad
        self.forget_bias_num = forget_bias_num

        self.AllW, self.d_AllW = None, None
        self.c0, self.d_c0 = None, None
        self.h0, self.d_h0 = None, None
        self.IFOGf = None
        self.IFOG = None
        self.Hin = None
        self.Ct = None
        self.C = None

    def connect_to(self, prev_layer=None):
        """Connection to the previous layer.

        Parameters
        ----------
        prev_layer : npdl.layers.Layer or None
            Previous layer.
        AllW : numpy.array
            ===== ==== === === ===
            type   i    f   o   g
            ----- ---- --- --- ---
            bias
            x2h
            h2h
            ===== ==== === === ===

        """
        super(BatchLSTM, self).connect_to(prev_layer)
        n_in = self.n_in
        n_out = self.n_out

        # init weights
        self.AllW = _zero((n_in + n_out + 1, 4 * n_out))

        # bias
        if self.forget_bias_num != 0:
            self.AllW[0, self.n_out: 2 * self.n_out] = self.forget_bias_num
        # Weights matrices for input x
        self.AllW[1:n_in + 1, n_out * 0:n_out * 1] = self.init((n_in, n_out))
        self.AllW[1:n_in + 1, n_out * 1:n_out * 2] = self.init((n_in, n_out))
        self.AllW[1:n_in + 1, n_out * 2:n_out * 3] = self.init((n_in, n_out))
        self.AllW[1:n_in + 1, n_out * 3:n_out * 4] = self.init((n_in, n_out))
        # Weights matrices for memory cell
        self.AllW[n_in + 1:, n_out * 0:n_out * 1] = self.inner_init((n_out, n_out))
        self.AllW[n_in + 1:, n_out * 1:n_out * 2] = self.inner_init((n_out, n_out))
        self.AllW[n_in + 1:, n_out * 2:n_out * 3] = self.inner_init((n_out, n_out))
        self.AllW[n_in + 1:, n_out * 3:n_out * 4] = self.inner_init((n_out, n_out))

    def forward(self, input, c0=None, h0=None):
        
        # print(f"input{input.shape}")
        # checking
        assert np.ndim(input) == 3, 'Only support batch training.'
        assert input.shape[2] == self.n_in

        # shape
        nb_batch, nb_seq, n_in = input.shape
        self.nb_batch = nb_batch
        self.nb_seq = nb_seq

        # data
        input = np.transpose(input, (1, 0, 2))
        self.c0 = _zero((nb_batch, self.n_out)) if c0 is None else c0
        self.h0 = _zero((nb_batch, self.n_out)) if h0 is None else h0

        # Perform the LSTM forward pass with X as the input #
        # x plus h plus bias, lol
        xphpb = self.AllW.shape[0]
        # input [1, xt, ht-1] to each tick of the LSTM
        Hin = _zero((nb_seq, nb_batch, xphpb))
        # hidden representation of the LSTM (gated cell content)
        Hout = _zero((nb_seq, nb_batch, self.n_out))
        # input, forget, output, gate (IFOG)
        IFOG = _zero((nb_seq, nb_batch, self.n_out * 4))
        # after nonlinearity
        IFOGf = _zero((nb_seq, nb_batch, self.n_out * 4))
        # cell content
        C = _zero((nb_seq, nb_batch, self.n_out))
        # tanh of cell content
        Ct = _zero((nb_seq, nb_batch, self.n_out))
        for t in range(nb_seq):
            # concat [x,h] as input to the LSTM
            prevh = Hout[t - 1] if t > 0 else self.h0
            # bias
            Hin[t, :, 0] = 1
            Hin[t, :, 1:n_in + 1] = input[t]
            Hin[t, :, n_in + 1:] = prevh
            # compute all gate activations. dots: (most work is this line)
            IFOG[t] = Hin[t].dot(self.AllW)
            # non-linearities
            # sigmoids; these are the gates
            IFOGf[t, :, :3 * self.n_out] = 1.0 / (1.0 + np.exp(-IFOG[t, :, :3 * self.n_out]))
            # tanh
            IFOGf[t, :, 3 * self.n_out:] = np.tanh(IFOG[t, :, 3 * self.n_out:])
            # compute the cell activation
            prevc = C[t - 1] if t > 0 else self.c0
            C[t] = IFOGf[t, :, :self.n_out] * IFOGf[t, :, 3 * self.n_out:] + \
                   IFOGf[t, :, self.n_out:2 * self.n_out] * prevc
            Ct[t] = np.tanh(C[t])
            Hout[t] = IFOGf[t, :, 2 * self.n_out:3 * self.n_out] * Ct[t]

        # record
        self.last_output = np.transpose(Hout, (1, 0, 2))
        self.IFOGf = IFOGf
        self.IFOG = IFOG
        self.Hin = Hin
        self.Ct = Ct
        self.C = C

        # print(f"last_output-{self.last_output.shape}")
        if self.return_sequence:
            return self.last_output
        else:
            return self.last_output[:, -1, :]

    def backward(self, pre_grad, dcn=None, dhn=None):
        """Backward propagation.
        
        Parameters
        ----------
        pre_grad : numpy.array
            Gradients propagated to this layer.
        dcn : numpy.array
            Gradients of cell state at `n` time step.
        dhn : numpy.array
            Gradients of hidden state at `n` time step.
            
        Returns
        -------
        numpy.array
            The gradients propagated to previous layer.
        """

        Hout = np.transpose(self.last_output, (1, 0, 2))
        nb_seq, batch_size, n_out = Hout.shape
        input_size = self.AllW.shape[0] - n_out - 1  # -1 due to bias

        self.d_AllW = _zero(self.AllW.shape)
        self.d_h0 = _zero((batch_size, n_out))

        # backprop the LSTM
        dIFOG = _zero(self.IFOG.shape)
        dIFOGf = _zero(self.IFOGf.shape)
        dHin = _zero(self.Hin.shape)
        dC = _zero(self.C.shape)
        layer_grad = _zero((nb_seq, batch_size, input_size))
        # make a copy so we don't have any funny side effects

        # prepare layer gradients
        if self.return_sequence:
            timesteps = list(range(nb_seq))[::-1]
            assert np.ndim(pre_grad) == 3
        else:
            timesteps = [nb_seq - 1]
            assert np.ndim(pre_grad) == 2
            tmp = _zero((self.nb_batch, self.nb_seq, self.n_out))
            tmp[:, -1, :] = pre_grad
            pre_grad = tmp
        dHout = np.transpose(pre_grad, (1, 0, 2)).copy()

        # carry over gradients from later
        if dcn is not None: dC[nb_seq - 1] += dcn.copy()
        if dhn is not None: dHout[nb_seq - 1] += dhn.copy()

        for t in timesteps:

            tanhCt = self.Ct[t]
            dIFOGf[t, :, 2 * n_out:3 * n_out] = tanhCt * dHout[t]
            # backprop tanh non-linearity first then continue backprop
            dC[t] += (1 - tanhCt ** 2) * (self.IFOGf[t, :, 2 * n_out:3 * n_out] * dHout[t])

            if t > 0:
                dIFOGf[t, :, n_out:2 * n_out] = self.C[t - 1] * dC[t]
                dC[t - 1] += self.IFOGf[t, :, n_out:2 * n_out] * dC[t]
            else:
                dIFOGf[t, :, n_out:2 * n_out] = self.c0 * dC[t]
                self.d_c0 = self.IFOGf[t, :, n_out:2 * n_out] * dC[t]
            dIFOGf[t, :, :n_out] = self.IFOGf[t, :, 3 * n_out:] * dC[t]
            dIFOGf[t, :, 3 * n_out:] = self.IFOGf[t, :, :n_out] * dC[t]

            # backprop activation functions
            dIFOG[t, :, 3 * n_out:] = (1 - self.IFOGf[t, :, 3 * n_out:] ** 2) * dIFOGf[t, :, 3 * n_out:]
            y = self.IFOGf[t, :, :3 * n_out]
            dIFOG[t, :, :3 * n_out] = (y * (1.0 - y)) * dIFOGf[t, :, :3 * n_out]

            # backprop matrix multiply
            self.d_AllW += np.dot(self.Hin[t].transpose(), dIFOG[t])
            dHin[t] = dIFOG[t].dot(self.AllW.transpose())

            # backprop the identity transforms into Hin
            layer_grad[t] = dHin[t, :, 1:input_size + 1]
            if t > 0:
                dHout[t - 1, :] += dHin[t, :, input_size + 1:]
            else:
                self.d_h0 += dHin[t, :, input_size + 1:]

        layer_grad = np.transpose(layer_grad, (1, 0, 2))
        return layer_grad

    @property
    def params(self):
        return [self.AllW, ]

    @property
    def grads(self):
        return [self.d_AllW, ]
