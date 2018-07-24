import numpy as np
from scipy.special import expit
import math

# Following are used for importing and converting sentence to vectors
import array
import collections
import io
import linecache
import re

# Following are used for training
import time

# ===================================================================== #
# ============================= Functions ============================= #
# ===================================================================== #

# -------------------------- init_dictionary -------------------------- #
# Function for initialilizing the dictionary given a file containing
# the words and their corresponding vectors
def init_dictionary(filename):
    dct = collections.OrderedDict()
    vectors = array.array('d')

    with io.open(filename, 'r', encoding='utf-8') as savefile:
        for i, line in enumerate(savefile):
            tokens = line.split(' ')
            word = tokens[0]
            entries = tokens[1:-1]
            dct[word] = list(float(x) for x in entries)

    # Initialize end of sentence token and store in dictionary
    dct['</eos>'] = [-0.1] * 300
    # Initialize unknown token and store in dictionary
    dct['</unk>'] = [0.1] * 300
    # Initialize acronym token and store in dictionary
    dct['</acr>'] = [-0.05] * 300
    # Initialize name token and store in dictionary
    dct['</name>'] = [0.05] * 300
    # Initialize empty token and store in dictionary
    dct['</empty>'] = [0] * 300

    # Return dictionary
    return dct

# ----------------------------- sigm ---------------------------------- #
# Returns the value of the sigmoid function evaluated at input z

def sigm(z):
    return expit(z)

# --------------------------- iof_gate -------------------------------- #
# Returns the value of the input gate, forget gate, or output gate
# given the following arrays:
#   Wx: weights for the input-input, input-forget, or input-output gate
#   Wh: weights for the hidden-input, hidden-forget, or hidden-output gate
#   Wc: weights for the cell-input, cell-forget, or cell-output gate
#   x : inputs at time t
#   h : hidden states at time t-1
#   c : cell states at time t-1
#   b : bias term for the input gate

def iof_gate(Wx, Wh, b, x, h):
    return sigm(np.dot(Wx,x) + np.dot(Wh,h) + b)

# -------------------------- cell_gate ------------------------------- #
# Returns the value of the cell state given the following arrays:
#   Wx: weights for the input-cell gate
#   Wh: weights for the hidden-cell gate
#   Wc: weights for the cell-output or cell-forget gate
#   f : forget states at time t
#   c : cell states at time t-1
#   i : input states at time t-1
#   x : inputs at time t
#   h : hidden states at time t-1
#   b : bias term for the cell gate

def cell_tilde(Wx, Wh, b, x, h):
    return np.tanh(np.dot(Wx,x) + np.dot(Wh,h) + b)

# -------------------------- cell_gate ------------------------------- #
# Returns the value of the cell state given the following arrays:
#   Wx: weights for the input-cell gate
#   Wh: weights for the hidden-cell gate
#   Wc: weights for the cell-output or cell-forget gate
#   f : forget states at time t
#   c : cell states at time t-1
#   i : input states at time t-1
#   x : inputs at time t
#   h : hidden states at time t-1
#   b : bias term for the cell gate

def cell_state(f, c, i, c_tilde):
#    return np.add(np.multiply(f,c), np.multiply(i, c_tilde))
    return (f*c + i*c_tilde)


# ------------------------- hidden_state ------------------------------ #
# Returns the value of the hidden state given the following arrays:
#   o : outputs at time t
#   c : cell states at time t

def hidden_state(o, c):
    return o*np.tanh(c)


# Routine for initializing the weight matrices 
def weight_init(Wrow):
    # The following returns a weight matrix with Wrow rows and Wcol
    # columns where the values are initialized using a Uniform
    # distribution between -0.08 and 0.08 (as in Seq. to Seq. paper)
    return np.random.uniform(-0.8, 0.8, Wrow) #0.05 * np.ones(Wrow)

# Routine for initializing the bias arrays
def bias_init(brow):
    # The following returns a bias vector with Wrow rows where the 
    # values are initialized using a Uniform distribution between 
    # -0.08 and 0.08 (as in Seq. to Seq. paper)
    return np.random.uniform(-0.8, 0.8, brow) #0.05 * np.ones(brow)


# ===================================================================== #
# ============================== Classes ============================== #
# ===================================================================== #

class Update_Direction:
    def __init__(self, Wrow_der, Wxcol_der, Whcol_der, brow_der, d_size, Decoder):
        # Initialize number of rows and columns for weight updates
        self.Wrow_der = Wrow_der
        self.Wxcol_der = Wxcol_der
        self.Whcol_der = Whcol_der
        self.brow_der = brow_der
        self.d_size = d_size
        self.Decoder = Decoder

        # Initialize values of directions used in updates of weights for 
        # loss function
        self.Wxi = np.zeros((self.Wrow_der, self.Wxcol_der))
        self.Whi = np.zeros((self.Wrow_der, self.Whcol_der))
        self.Wxf = np.zeros((self.Wrow_der, self.Wxcol_der))
        self.Whf = np.zeros((self.Wrow_der, self.Whcol_der))
        self.Wxo = np.zeros((self.Wrow_der, self.Wxcol_der))
        self.Who = np.zeros((self.Wrow_der, self.Whcol_der))
        self.Wxc = np.zeros((self.Wrow_der, self.Wxcol_der))
        self.Whc = np.zeros((self.Wrow_der, self.Whcol_der))

        self.bi = np.zeros(self.brow_der)
        self.bf = np.zeros(self.brow_der)
        self.bc = np.zeros(self.brow_der)
        self.bo = np.zeros(self.brow_der)

        # Check if decoder model is in use
        if (self.Decoder):
            self.b_mu = np.zeros(self.d_size)


    # Routine to reset update directions for Weights and bias terms
    def reset_directions(self):
        self.Wxi = np.zeros((self.Wrow_der, self.Wxcol_der))
        self.Whi = np.zeros((self.Wrow_der, self.Whcol_der))
        self.Wxf = np.zeros((self.Wrow_der, self.Wxcol_der))
        self.Whf = np.zeros((self.Wrow_der, self.Whcol_der))
        self.Wxo = np.zeros((self.Wrow_der, self.Wxcol_der))
        self.Who = np.zeros((self.Wrow_der, self.Whcol_der))
        self.Wxc = np.zeros((self.Wrow_der, self.Wxcol_der))
        self.Whc = np.zeros((self.Wrow_der, self.Whcol_der))

        self.bi = np.zeros(self.brow_der)
        self.bf = np.zeros(self.brow_der)
        self.bc = np.zeros(self.brow_der)
        self.bo = np.zeros(self.brow_der)

        # Check if decoder model is in use
        if (self.Decoder):
            self.b_mu = np.zeros(self.d_size)


    # Routine to print directions
    def print_directions(self):
        print("\n ================================================"
              "==============================")
        print(" ============================== Current Direcs "
              "===============================")
        print(" ================================================"
              "==============================")
        print("\n ------------------------------ Forget Direcs "
              "--------------------------------")
        print(" Whf = ", self.Whf)
        print(" Wxf = ", self.Wxf)
        print(" bf = ", self.bf)

        print("\n ------------------------------ Output Direcs "
              "--------------------------------")
        print(" Who = ", self.Who)
        print(" Wxo = ", self.Wxo)
        print(" bo = ", self.bo)

        print("\n ------------------------------- Input Direcs "
              "--------------------------------")
        print(" Whi = ", self.Whi)
        print(" Wxi = ", self.Wxi)
        print(" bi = ", self.bi)

        print("\n ------------------------------- Cell Direcs "
              "---------------------------------")
        print(" Whc = ", self.Whc)
        print(" Wxc = ", self.Wxc)
        print(" bc = ", self.bc, "\n")


class Parm:
    def __init__(self, n_x, x_dim, io_cols, n_time, learning_rate, 
                 dict_filename, num_samples, Decoder, Save_Init_Weights, 
                 Testing, USER_INPUT):

        # Initialize dictionary
        self.d = init_dictionary(dict_filename)
        self.d_size = len(self.d)

        # Initialize array to store tokens for a given sample and tokens
        # which have special significance (names, acronyms, and unknowns)
        self.tokens = []
        self.names = []
        self.acr = []
        self.unk = []

        # Initialize boolean Decoder which is: True if using decoder model
        #                                      False if using encoder model
        self.Decoder = Decoder

        # Initialize boolean Decoder which is: True if saving initial weights
        #                                      False otherwise
        self.Save_Initial_Weights = Save_Init_Weights

        # Initialize boolean Testing which is: True if testing
        #                                      False if training
        self.Testing = Testing

        # Initialize boolean which: True if user input sentence for translation
        #                           False otherwise
        self.USER_INPUT = USER_INPUT

        # Initialize input/output dimension
        self.io_col = io_cols

        # Initialize number of timesteps
        self.time = n_time

        # Initialize number of input arrays at each node
        self.num_inputs = n_x

        # Initialize dimension of each input array
        self.x_dim = x_dim
        self.x_dim_sum = np.sum(self.x_dim)

        # Initialize learning rate
        self.lr = learning_rate

        # Initialize index of current sample
        self.sample_index = 0

        # Initialze current token which stores current token in translation
        self.current_token = ''

        # Set number of rows and columns of weight arrays
        self.Wrow = self.io_col
        self.Wxcol = self.x_dim_sum
        self.Whcol = self.io_col
        self.brow = self.io_col
        self.Wrow_der = self.Wrow
        self.Wxcol_der = self.Wxcol
        self.Whcol_der = self.Whcol
        self.brow_der = self.brow

        # Initialize matrix of ones used in update of dE/dWx's
        self.Ones_Wx = np.ones((self.Wrow, self.Wxcol))

        # Initialize array for input, forget, output, cell, and hidden states
        self.i = np.zeros((self.time, self.io_col))
        self.f = np.zeros((self.time, self.io_col))
        self.o = np.zeros((self.time, self.io_col))
        self.c_tilde = np.zeros((self.time, self.io_col))
        self.c = np.zeros((self.time, self.io_col))
        self.h = np.zeros((self.time, self.io_col))

        self.cn1 = np.zeros(self.io_col)
        self.hn1 = np.zeros(self.io_col)

        # Initialize feature values
        self.x = np.zeros((self.time, self.x_dim_sum))

        # Initialize target valaues
        self.target = np.zeros((self.time, self.io_col))

        # Initialize weight arrays
        self.Wxi = weight_init((self.Wrow, self.Wxcol))
        self.Whi = weight_init((self.Wrow, self.Whcol))
        self.Wxf = weight_init((self.Wrow, self.Wxcol))
        self.Whf = weight_init((self.Wrow, self.Whcol))
        self.Wxo = weight_init((self.Wrow, self.Wxcol))
        self.Who = weight_init((self.Wrow, self.Whcol))
        self.Wxc = weight_init((self.Wrow, self.Wxcol))
        self.Whc = weight_init((self.Wrow, self.Whcol))

        # Initialize bias arrays
        self.bi = bias_init(self.brow)
        self.bf = bias_init(self.brow)
        self.bc = bias_init(self.brow)
        self.bo = bias_init(self.brow)

        # Initialize values of partial derivatives of loss function
        self.dE_dWxi = np.zeros((self.Wrow_der, self.Wxcol_der))
        self.dE_dWhi = np.zeros((self.Wrow_der, self.Whcol_der))
        self.dE_dWxf = np.zeros((self.Wrow_der, self.Wxcol_der))
        self.dE_dWhf = np.zeros((self.Wrow_der, self.Whcol_der))
        self.dE_dWxo = np.zeros((self.Wrow_der, self.Wxcol_der))
        self.dE_dWho = np.zeros((self.Wrow_der, self.Whcol_der))
        self.dE_dWxc = np.zeros((self.Wrow_der, self.Wxcol_der))
        self.dE_dWhc = np.zeros((self.Wrow_der, self.Whcol_der))

        self.dE_dbi = np.zeros(self.brow_der)
        self.dE_dbf = np.zeros(self.brow_der)
        self.dE_dbc = np.zeros(self.brow_der)
        self.dE_dbo = np.zeros(self.brow_der)

        # Initialize arrays used in update of dE_dWh and dE_dContext arrays
        self.dh_dWhi = np.zeros((self.Wrow_der, self.Whcol_der))
        self.dh_dWhf = np.zeros((self.Wrow_der, self.Whcol_der))
        self.dh_dWho = np.zeros((self.Wrow_der, self.Whcol_der))
        self.dh_dWhc = np.zeros((self.Wrow_der, self.Whcol_der))
            # dc_dContext Actually will end up the size as above but to 
            # save on computational cost we initialize at this size
            #self.dc_dContext = np.zeros(self.Wrow_der)
        self.sum_dE_dWhi = np.zeros((self.Wrow_der, self.Whcol_der))
        self.sum_dE_dWhf = np.zeros((self.Wrow_der, self.Whcol_der))
        self.sum_dE_dWho = np.zeros((self.Wrow_der, self.Whcol_der))
        self.sum_dE_dWhc = np.zeros((self.Wrow_der, self.Whcol_der))
        self.com_sum_dWh = np.zeros((self.Wrow_der, self.Whcol_der))
            # com_sum_dContext Actually will end up the size as above but to 
            # save on computational cost we initialize at this size
            #self.com_sum_dContext = np.zeros(self.Wrow_der)

        # Initialize update directions for model
        self.dir = Update_Direction(self.Wrow_der, self.Wxcol_der, 
                                    self.Whcol_der, self.brow_der, 
                                    self.d_size, self.Decoder)

        # Check if encoder model is requested
        if (not self.Decoder):
            self.dE_dDecoder = np.zeros((self.Wrow_der, self.Wxcol_der))

            # Check if initial weights are to be saved
            if (self.Save_Initial_Weights):
                # Print inital weights to file
                self.save_weights_to_file(False)
            # Otherwise load the already saved initial weights
            else:
                # Load initial weights from file
                self.load_weights_from_file(self.Testing)   

        # Check if decoder model is requested
        if (self.Decoder):
            # Initialize dictionary matrix used in update of softmax gradient
            self.D_Matrix = np.zeros((self.d_size, self.io_col))
            items = list(self.d.items())            

            for i in range(self.d_size):
                self.D_Matrix[i] = np.copy(items[i][1])

            # Initialize number of columns for the context weight matrix
            self.WContext = self.x_dim[0]

            # Initialize values of partial derivatives of loss function
            self.dE_dh = np.zeros((self.time, self.Whcol_der))
            self.dE_dContext = np.zeros((self.Wrow_der, self.WContext))
            self.sum_dh_dContext = np.zeros((self.Wrow_der, self.WContext))
            self.com_sum_dContext = np.zeros((self.Wrow_der, self.WContext))
            self.dE_db_mu = np.zeros(self.d_size)

            # Initialize arrays used in update of dE_dWh and dE_dContext arrays
            self.dh_dContext = np.zeros((self.Wrow_der, self.WContext))

            # Initialize data set size N from density function
            self.N_samples = num_samples

            # Initialize array for mu output values (from softmax function)
            self.mu = np.zeros(self.d_size)

            # Initialize bias array for softmax function
            self.b_mu = bias_init(self.d_size)

            # Initialize array for density function output values
            self.density = np.zeros(self.time)

            # Initialize total density
            self.total_density = 0
        
            # Initialize context vector
            self.context = np.zeros(self.WContext)

            # Check if initial weights are to be saved
            if (self.Save_Initial_Weights):
                # Print inital weights to file
                self.save_weights_to_file(False)
            # Otherwise load the already saved initial weights
            else:
                # Load initial weights from file
                self.load_weights_from_file(self.Testing)   


    # ----------------------- Printing Routines ----------------------- #
    def print_inputs(self):
        print("\n ------------------------------- Current Inputs "
              "-------------------------------")
        print(" Current input = ", self.x, "\n")

    def print_errs(self):
        print("\n ------------------------------- Current Errors "
              "-------------------------------")
        print(" err = ", self.err)
        print(" Total error = ", self.totalErr, "\n")

    def print_total_err(self):
        print("\n -------------------------------- Total Error "
              "--------------------------------")
        print(" Total error = ", self.totalErr, "\n")

    def print_states(self):
        print("\n ------------------------------- Current States "
              "-------------------------------")
        print(" i = ", self.i)
        print(" f = ", self.f)
        print(" o = ", self.o)
        print(" c̃ = ", self.c_tilde)
        print(" c = ", self.c)
        print(" h = ", self.h, "\n")

    def print_weights(self):
        print("\n ================================================"
              "==============================")
        print(" ============================== Current Weights "
              "===============================")
        print(" ================================================"
              "==============================")
        print("\n ------------------------------ Forget Weights "
              "--------------------------------")
        print(" Whf = ", self.Whf)
        print(" Wxf = ", self.Wxf)
        print(" bf = ", self.bf)

        print("\n ------------------------------ Output Weights "
              "--------------------------------")
        print(" Who = ", self.Who)
        print(" Wxo = ", self.Wxo)
        print(" bo = ", self.bo)

        print("\n ------------------------------- Input Weights "
              "--------------------------------")
        print(" Whi = ", self.Whi)
        print(" Wxi = ", self.Wxi)
        print(" bi = ", self.bi)

        print("\n ------------------------------- Cell Weights "
              "---------------------------------")
        print(" Whc = ", self.Whc)
        print(" Wxc = ", self.Wxc)
        print(" bc = ", self.bc, "\n")

    def save_weights_to_file(self, FINAL):
        # Check if initial weights are to be saved (for training purposes)
        if (not FINAL):
            # Check if encoder model in use
            if (not self.Decoder):
                np.save('encoder_Whf', self.Whf)
                np.save('encoder_Wxf', self.Wxf)
                np.save('encoder_bf', self.bf)
                np.save('encoder_Who', self.Who)
                np.save('encoder_Wxo', self.Wxo)
                np.save('encoder_bo', self.bo)
                np.save('encoder_Whi', self.Whi)
                np.save('encoder_Wxi', self.Wxi)
                np.save('encoder_bi', self.bi)
                np.save('encoder_Whc', self.Whc)
                np.save('encoder_Wxc', self.Wxc)
                np.save('encoder_bc', self.bc)
            # Otherwise decoder model is in use
            else:
                np.save('decoder_Whf', self.Whf)
                np.save('decoder_Wxf', self.Wxf)
                np.save('decoder_bf', self.bf)
                np.save('decoder_Who', self.Who)
                np.save('decoder_Wxo', self.Wxo)
                np.save('decoder_bo', self.bo)
                np.save('decoder_Whi', self.Whi)
                np.save('decoder_Wxi', self.Wxi)
                np.save('decoder_bi', self.bi)
                np.save('decoder_Whc', self.Whc)
                np.save('decoder_Wxc', self.Wxc)
                np.save('decoder_bc', self.bc)       
                np.save('decoder_b_mu', self.b_mu)  
        # Otherwise final weights are to be saved (at end of training so 
        # they can be used for testing purposes)
        else:
            # Check if encoder model in use
            if (not self.Decoder):
                np.save('final_encoder_Whf', self.Whf)
                np.save('final_encoder_Wxf', self.Wxf)
                np.save('final_encoder_bf', self.bf)
                np.save('final_encoder_Who', self.Who)
                np.save('final_encoder_Wxo', self.Wxo)
                np.save('final_encoder_bo', self.bo)
                np.save('final_encoder_Whi', self.Whi)
                np.save('final_encoder_Wxi', self.Wxi)
                np.save('final_encoder_bi', self.bi)
                np.save('final_encoder_Whc', self.Whc)
                np.save('final_encoder_Wxc', self.Wxc)
                np.save('final_encoder_bc', self.bc)
            # Otherwise decoder model is in use
            else:
                np.save('final_decoder_Whf', self.Whf)
                np.save('final_decoder_Wxf', self.Wxf)
                np.save('final_decoder_bf', self.bf)
                np.save('final_decoder_Who', self.Who)
                np.save('final_decoder_Wxo', self.Wxo)
                np.save('final_decoder_bo', self.bo)
                np.save('final_decoder_Whi', self.Whi)
                np.save('final_decoder_Wxi', self.Wxi)
                np.save('final_decoder_bi', self.bi)
                np.save('final_decoder_Whc', self.Whc)
                np.save('final_decoder_Wxc', self.Wxc)
                np.save('final_decoder_bc', self.bc)       
                np.save('final_decoder_b_mu', self.b_mu)  

    def load_weights_from_file(self, FINAL):
        # Check if initial weights are to be used (for training purposes)
        if (not FINAL):
            # Check if encoder model in use
            if (not self.Decoder):
                self.Whf = np.load('encoder_Whf.npy')
                self.Wxf = np.load('encoder_Wxf.npy')
                self.bf = np.load('encoder_bf.npy')
                self.Who = np.load('encoder_Who.npy')
                self.Wxo = np.load('encoder_Wxo.npy')
                self.bo = np.load('encoder_bo.npy')
                self.Whi = np.load('encoder_Whi.npy')
                self.Wxi = np.load('encoder_Wxi.npy')
                self.bi = np.load('encoder_bi.npy')
                self.Whc = np.load('encoder_Whc.npy')
                self.Wxc = np.load('encoder_Wxc.npy')
                self.bc = np.load('encoder_bc.npy')
            # Otherwise decoder model is in use
            else:
                self.Whf = np.load('decoder_Whf.npy')
                self.Wxf = np.load('decoder_Wxf.npy')
                self.bf = np.load('decoder_bf.npy')
                self.Who = np.load('decoder_Who.npy')
                self.Wxo = np.load('decoder_Wxo.npy')
                self.bo = np.load('decoder_bo.npy')
                self.Whi = np.load('decoder_Whi.npy')
                self.Wxi = np.load('decoder_Wxi.npy')
                self.bi = np.load('decoder_bi.npy')
                self.Whc = np.load('decoder_Whc.npy')
                self.Wxc = np.load('decoder_Wxc.npy')
                self.bc = np.load('decoder_bc.npy')
                self.b_mu = np.load('decoder_b_mu.npy')

        # Otherwise final weights are to be used for testing
        else:
            # Check if encoder model in use
            if (not self.Decoder):
                self.Whf = np.load('final_encoder_Whf.npy')
                self.Wxf = np.load('final_encoder_Wxf.npy')
                self.bf = np.load('final_encoder_bf.npy')
                self.Who = np.load('final_encoder_Who.npy')
                self.Wxo = np.load('final_encoder_Wxo.npy')
                self.bo = np.load('final_encoder_bo.npy')
                self.Whi = np.load('final_encoder_Whi.npy')
                self.Wxi = np.load('final_encoder_Wxi.npy')
                self.bi = np.load('final_encoder_bi.npy')
                self.Whc = np.load('final_encoder_Whc.npy')
                self.Wxc = np.load('final_encoder_Wxc.npy')
                self.bc = np.load('final_encoder_bc.npy')
            # Otherwise decoder model is in use
            else:
                self.Whf = np.load('final_decoder_Whf.npy')
                self.Wxf = np.load('final_decoder_Wxf.npy')
                self.bf = np.load('final_decoder_bf.npy')
                self.Who = np.load('final_decoder_Who.npy')
                self.Wxo = np.load('final_decoder_Wxo.npy')
                self.bo = np.load('final_decoder_bo.npy')
                self.Whi = np.load('final_decoder_Whi.npy')
                self.Wxi = np.load('final_decoder_Wxi.npy')
                self.bi = np.load('final_decoder_bi.npy')
                self.Whc = np.load('final_decoder_Whc.npy')
                self.Wxc = np.load('final_decoder_Wxc.npy')
                self.bc = np.load('final_decoder_bc.npy')
                self.b_mu = np.load('final_decoder_b_mu.npy')


    def print_partials(self):
        print("\n ================================================"
              "==============================")
        print(" ============================== Current Partials "
              "==============================")
        print(" ================================================"
              "==============================")
        print("\n ------------------------ With respect to Forget Gate "
              "-------------------------")
        print(" dE/dWhf = ", self.dE_dWhf)
        print(" dE/dWxf = ", self.dE_dWxf)
        print(" dh/dWhf = ", self.dh_dWhf)
        print(" dE/dbf = ", self.dE_dbf)

        print("\n ------------------------ With respect to Output Gate "
              "-------------------------")
        print(" dE/dWho = ", self.dE_dWho)
        print(" dE/dWxo = ", self.dE_dWxo)
        print(" dE/dbo = ", self.dE_dbo)

        print("\n ------------------------ With respect to Input Gate "
              "--------------------------")
        print(" dE/dWhi = ", self.dE_dWhi)
        print(" dE/dWxi = ", self.dE_dWxi)
        print(" dE/dbi = ", self.dE_dbi)

        print("\n ------------------------ With respect to Cell Gate "
              "---------------------------")
        print(" dE/dWhc = ", self.dE_dWhc)
        print(" dE/dWxc = ", self.dE_dWxc)
        print(" dE/dbc = ", self.dE_dbc, "\n")

    # ------------------------ Reset Routines ------------------------- #
    # Routine to reset the cell states so that they are set to the 
    # updated number of timesteps
    def states_reset(self):
        self.i = np.zeros((self.time, self.io_col))
        self.f = np.zeros((self.time, self.io_col))
        self.o = np.zeros((self.time, self.io_col))
        self.c_tilde = np.zeros((self.time, self.io_col))
        self.c = np.zeros((self.time, self.io_col))
        self.h = np.zeros((self.time, self.io_col))

        if (self.Decoder):
            # Initialize array for density function output values
            self.density = np.zeros(self.time)
            self.dE_dh = np.zeros((self.time, self.Whcol_der))

    # Routine to reset derivative values and arrays used in derivatives
    def der_reset(self):
        self.dE_dWxi = np.zeros((self.Wrow_der, self.Wxcol_der))
        self.dE_dWhi = np.zeros((self.Wrow_der, self.Whcol_der))
        self.dE_dWxf = np.zeros((self.Wrow_der, self.Wxcol_der))
        self.dE_dWhf = np.zeros((self.Wrow_der, self.Whcol_der))
        self.dE_dWxo = np.zeros((self.Wrow_der, self.Wxcol_der))
        self.dE_dWho = np.zeros((self.Wrow_der, self.Whcol_der))
        self.dE_dWxc = np.zeros((self.Wrow_der, self.Wxcol_der))
        self.dE_dWhc = np.zeros((self.Wrow_der, self.Whcol_der))
        self.dE_dbi = np.zeros(self.brow_der)
        self.dE_dbf = np.zeros(self.brow_der)
        self.dE_dbc = np.zeros(self.brow_der)
        self.dE_dbo = np.zeros(self.brow_der)

        # Reset arrays used in update of dE_dWh arrays
        self.dh_dWhi = np.zeros((self.Wrow_der, self.Whcol_der))
        self.dh_dWhf = np.zeros((self.Wrow_der, self.Whcol_der))
        self.dh_dWho = np.zeros((self.Wrow_der, self.Whcol_der))
        self.dh_dWhc = np.zeros((self.Wrow_der, self.Whcol_der))
            # dh_dContext Actually will end up the size as above but to 
            # save on computational cost we initialize at this size
            #self.dh_dContext = np.zeros(self.Wrow_der)
        self.sum_dE_dWhi = np.zeros((self.Wrow_der, self.Whcol_der))
        self.sum_dE_dWhf = np.zeros((self.Wrow_der, self.Whcol_der))
        self.sum_dE_dWho = np.zeros((self.Wrow_der, self.Whcol_der))
        self.sum_dE_dWhc = np.zeros((self.Wrow_der, self.Whcol_der))
        self.com_sum_dWh = np.zeros((self.Wrow_der, self.Whcol_der))
            # com_sum_dContext Actually will end up the size as above but to 
            # save on computational cost we initialize at this size
            #self.com_sum_dContext = np.zeros(self.Wrow_der)

        # Reset array used exclusively in decoder
        if (self.Decoder):
            self.dE_dContext = np.zeros((self.Wrow_der, self.WContext))
            self.dh_dContext = np.zeros((self.Wrow_der, self.WContext))
            self.sum_dh_dContext = np.zeros((self.Wrow_der, self.WContext))
            self.com_sum_dContext = np.zeros((self.Wrow_der, self.WContext))

    def dE_dh_reset(self):
        self.dE_dh = np.zeros((self.time, self.Whcol_der))
        self.dE_db_mu = np.zeros(self.d_size)

    # Routine to reset derivative values and arrays used in derivatives
    def der_reset_common_sum(self):
        self.com_sum_dWh = np.zeros((self.Wrow_der, self.Whcol_der))

        if (self.Decoder):
            self.com_sum_dContext = np.zeros((self.Wrow_der, self.WContext))

    # ----------------------- Updating Routines ----------------------- #

    # ------------------------ sentence_to_matrix ------------------------- #
    # This command will pick the nth row out of a text file and return 
    # the following:
    #       1. Array containing the tokens from the input sentence
    #       2. matrix in which the ith row is a vector corresponding to the 
    #          ith word in the tokens array from the nth row of the text file
    def sentence_to_matrix(self, sample_filename, sample_index):
        # Reset all token arrays in class
        self.tokens = []
        self.names = []
        self.acr = []
        self.unk = []

        # Get nth line from text file
        if (not self.USER_INPUT):
            line = linecache.getline(sample_filename, sample_index)
        else:
            line = sample_filename
    
        # Remove the endline character, \n, from the sentence. This is done since
        # \n is not stored in the dictionary
        line = line.rsplit('\n', 1)

        # Split the sentence into individual tokens based on where spaces occur
        # We use line[0] here because line[1] contains '' from when we removed 
        # the endline character \n
        line = line[0]
        p = re.compile(r'(\W+)')        
        self.tokens = p.split(line)

        # Strip all whitespace in tokens so they will match with dictionary tokens
        self.tokens = [x.strip(' ') for x in self.tokens]

        # Remove all empty tokens from tokens list
        self.tokens = list(filter(None, self.tokens))\

        # Add the end of input token to the tokens array
        self.tokens.append('</eos>')
    
        # Initialize number of words in current sentence
        num_words = len(self.tokens)

        # Update number of timesteps for this sample
        self.time = num_words
    
        # Initialize matrix to store sentence as rows of vectors. We increase 
        # the number of rows by one to add an end of input character.
        self.x = np.zeros((num_words, self.io_col))

        for i in range(0, self.time): 
            # Convert each token to lowercase so it can be found in the dictionary
            temp_str = self.tokens[i]
    
            # Check if ith token is empty character ''
            if(len(temp_str) == 0):
                # Update the ith token to be the empty token '</empty>'
                self.tokens[i] = '</empty>'
                # Set the ith row of the matrix to be the vector corresponding to
                # the ith token in the sentence    
                self.x[i] = np.copy(self.d['</empty>'])            

            # Check if ith token is in dictionary
            elif(temp_str in self.d):
                # Set the ith row of the matrix to be the vector corresponding to
                # the ith token in the sentence
                self.x[i] = np.copy(self.d[temp_str])
    
            # Check if lowercase form of ith token is in dictionary
            elif(temp_str.lower() in self.d):
                # Store the lowercase form of the ith token in the tokens array
                self.tokens[i] = temp_str.lower()
                # Set the ith row of the matrix to be the vector corresponding to
                # the ith token in the sentence
                self.x[i] = np.copy(self.d[self.tokens[i]])
    
            # Check if entire word is capitalized (susptected to be acronym)
            elif(temp_str.isupper()):
                # Store the original suspected acronym in the acr array
                self.acr.append(temp_str)
                # Update the ith token to be the acronym token '</acr>'
                self.tokens[i] = '</acr>'
                # Set the ith row of the matrix to be the vector corresponding to
                # the acronym token '</acr>'
                self.x[i] = np.copy(self.d['</acr>'])
    
            # Check if just the first letter is capitalized (susptected to be name)
            elif(temp_str[0].isupper()):
                # Store the original suspected name in the name array
                self.names.append(temp_str)
                # Update the ith token to be the name token '</name>'
                self.tokens[i] = '</name>'
                # Set the ith row of the matrix to be the vector corresponding to
                # the acronym token '</name>'
                self.x[i] = np.copy(self.d['</name>'])
    
            # Otherwise we consider the word to be unknown
            else:
                # Append the unknown word to the unk array
                self.unk.append(temp_str)
                # Update the ith token to be the unknown token '</unk>'
                self.tokens[i] = '</unk>'
                # Set the ith row of the matrix to be the vector corresponding to
                # the acronym token '</acr>'
                self.x[i] = np.copy(self.d['</unk>'])


    # Routine to update sample index, feature, and target
    def io_update(self, x_new, t_new, index):
        self.sample_index = index
        self.x = np.copy(x_new)
        self.target = np.copy(t_new)

    # Routine to update tokens from sentence and corresponding matrix
    def input_update(self, sample_filename, sample_index, context_vector):
        # Check if using decoder model
        if (self.Decoder):
#            self.hn1 = np.tanh(context_vector)
            # Check if not translating user input sentence
            if (not self.USER_INPUT):
                # Update self.time, self.tokens, and self.x based on 
                # given sample
                self.sentence_to_matrix(sample_filename, sample_index)
            # Create matrix with self.time rows all of which are 
            # all the context_vector
            context_matrix = np.array([context_vector,] * self.time)
            # Contatenate the rows of the context matrix together with the 
            # word vectors from the target sentence
            if (not self.USER_INPUT):
                self.x = np.copy(np.hstack((context_matrix, self.x)))
            else:
                temp_matrix = np.array([np.zeros(self.io_col),] * self.time)
                self.x = np.copy(np.hstack((context_matrix, temp_matrix)))
        else:
            # Update self.time, self.tokens, and self.x based on given sample
            self.sentence_to_matrix(sample_filename, sample_index)

    # Routine to update last 300 columns of entry t of matrix x
    def test_x_update(self, t):
        # Check if t > 0
        if (t > 0):
            self.x[t, 300:600] = np.copy(self.h[t-1])
        else:
            self.x[t, 300:600] = np.copy(np.zeros(self.io_col))

    # ---------------------------- softmax -------------------------------- #
    # Returns the array of softmax values over a given array z and also
    # returns the derivative of the softmax function if the boolean 
    # variable COMPUTE_DERIV is set to True. This method is used for 
    # numerical stability and to prevent overflow when computing the 
    # exponential function.

    def softmax(self, z, index, t, COMPUTE_DERIV):
        # For every component of z, compute e^(z_i - max(z))
        e = np.exp(z - np.max(z))
    
        # Check if the derivative has also been requested
        if (COMPUTE_DERIV):
            # Compute common sum found in softmax and its derivative
            # s = sum_{i = 1}^{size(z)} e^(z_i - max(z))
            s = np.sum(e)
        
            # Compute array of softmax values
            # temp = array of softmax values for vector z
            temp = e/s
            self.mu[t] = np.copy(temp[index])

            # Compute gradient of softmax
            # Initialize vector corresponding to token at current timestep
            token_vec = np.copy(self.d[self.tokens[t]])
            # Initialize sum of vectors and index
            vec_sum = np.zeros(self.Wrow)
            # Compute sum_{j = 1}^{size(dictionary)} e^z (token_vec - token_j)
            temp_matrix = np.subtract(token_vec, self.D_Matrix)
            vec_sum = np.copy(np.dot(temp_matrix.T, e))

            # Note that dE/dh = dE/dmu * dmu/dh where 
            # dE/dmu = -1/(mu * N_samples) and dmu/dh = mu * (vec_sum/s). 
            # So here we store the vector dE/dh = temp/(-N_samples * s) 
            # instead of computing the two partials separately and multiplying
            s *= - self.N_samples
            self.dE_dh[t] = vec_sum/s  #self.mu[t] * (vec_sum/s)

            # Note that dE/db_mu = dE/dmu * dmu/db_mu where 
            # dE/dmu = -1/(mu * N_samples), dmu/db_mu[i] = - mu e^(z_i) for all
            # i not corresponding to the current token and 
            # dmu/db_mu = mu * (1 - mu) for the current token. So here we add 
            # these values to the vector dE/db_mu = temp/(-N_samples * s) 
            # instead of computing the two partials separately and multiplying
            e = np.exp(z)
            e[index] = (self.mu[t] - 1)
            np.add(self.dE_db_mu, e/self.N_samples, self.dE_db_mu)

        else:
            temp = e/np.sum(e)

            # Check if training
            if (not self.Testing):
                self.mu[t] = temp[index]

            # Otherwise testing is taking place
            else:
                token_index = np.argmax(e)
                items = list(self.d.items())
                self.current_token = items[token_index][0]
                print(self.current_token, end=' ')
                


    # Routine to compute softmax at a given timestep given matrix W containing
    # the words of the dictionary in the target language
    def mu_update(self, t):
        # Check if training
        if (not self.Testing):
            # Initialize index of tokens[t] in the ordered dictionary
            token_index = list(self.d.keys()).index(self.tokens[t])
    
            # For every key in the dictionary append the dot product of 
            # the word vector with the output from LSTM node
            # Initialize vector containing temporary token vector
            temp = np.dot(self.D_Matrix, self.h[t])

            # Add the bias term corresponding to tokens[t] to each 
            # component of temp
            np.add(temp, self.b_mu, temp)

            # Compute the softmax function value at the softmax index and
            # store the gradient of the softmax function for the backward
            # propagation phase in order to save computation time
            self.softmax(temp, token_index, t, True)
            # Update mu[t] to be the softmax value corresponding to tokens[t]

        # Otherwise testing is taking place
        else:
            # For every key in the dictionary append the dot product of 
            # the word vector with the output from LSTM node
            # Initialize vector containing temporary token vector
            temp = np.dot(self.D_Matrix, self.h[t])

            # Add the bias term corresponding to tokens[t] to each 
            # component of temp
            np.add(temp, self.b_mu, temp)

            # Compute the softmax function value at the softmax index and
            # store the gradient of the softmax function for the backward
            # propagation phase in order to save computation time
            self.softmax(temp, 0, t, False)


    # Routine to update the state values in the LSTM at a single timestep
    def cell_state_update(self, t):
        # Check value of t
        if (t == 0):
            # Update input state at timestep 0
            self.i[0] = iof_gate(self.Wxi, self.Whi, self.bi, self.x[0], 
                                 self.hn1)

            # Update forget state at timestep 0
            self.f[0] = iof_gate(self.Wxf, self.Whf, self.bf, self.x[0], 
                                 self.hn1)

            # Update output state at timestep 0
            self.o[0] = iof_gate(self.Wxo, self.Who, self.bo, self.x[0], 
                                 self.hn1)

            # Update cell tilde at timestep 0
            self.c_tilde[0] = cell_tilde(self.Wxc, self.Whc, self.bc, 
                                         self.x[0], self.hn1)

            # Update cell state at timestep 0
            self.c[0] = cell_state(self.f[0], self.cn1, self.i[0], 
                                   self.c_tilde[0])

            # Update hidden state at timestep 0
            self.h[0] = hidden_state(self.o[0], self.c[0])

        else:
            # Update input state at timestep t
            self.i[t] = iof_gate(self.Wxi, self.Whi, self.bi, self.x[t], 
                                 self.h[t-1])

            # Update forget state at timestep t
            self.f[t] = iof_gate(self.Wxf, self.Whf, self.bf, self.x[t], 
                                 self.h[t-1])

            # Update output state at timestep t
            self.o[t] = iof_gate(self.Wxo, self.Who, self.bo, self.x[t], 
                                 self.h[t-1])

            # Update cell tilde at timestep t
            self.c_tilde[t] = cell_tilde(self.Wxc, self.Whc, self.bc, 
                                         self.x[t], self.h[t-1])

            # Update cell state at timestep t
            self.c[t] = cell_state(self.f[t], self.c[t-1], self.i[t], 
                                   self.c_tilde[t])

            # Update hidden state at timestep t
            self.h[t] = hidden_state(self.o[t], self.c[t])

    # ------------------ Routines for supervised learning ----------------- #    
    # Routine to update the density value in the decoder at a single timestep
    def err_update(self, t):
        #Alternate code: 
        np.subtract(self.h[t], self.target[t], self.err[t])

    # Routine to compute the total error in the LSTM
    def totalErr_update(self):
        # Initialize temp
        temp = 0

        # Compute total error
        for i in range(self.time):
            temp += 0.5 * np.dot(self.err[i], self.err[i])

        # Store total error
        self.totalErr = temp

    # ------------------ Routines for density estimation ------------------ #    
    # Routine to update the density value in the decoder at a single timestep
    def density_update(self, t):
        self.density[t] = - np.log(self.mu[t])/self.N_samples

    # Routine to compute the total density for the decoder at a single sample
    def total_density_update(self):
        self.total_density = np.sum(self.density)

    # Routine to update the partial derivative for the encoder based on the
    # partial derivative of the loss function value from the decoder
    def decoder_to_encoder_partial_update(self, Mat):
        self.dE_dDecoder = np.copy(Mat)

    # ------------------------- common_del_f ------------------------------ #
    # Returns the array which is the common multiple in dE_dWxf and dE_dbf 
    # given the following input:
    #   t : current timestep

    def common_del_f(self, t, X_OR_B):
        # Initialize temp arrays
        u = np.zeros(self.io_col)
        v = np.zeros(self.io_col)

        # v = (1 - f_t)
        np.subtract(np.ones(self.io_col), self.f[t], v)
        # u = f_t * (1 - f_t)
        np.multiply(self.f[t], v, u)

        # Check if timestep > 0
        if (t > 0):
            # u = f_t * (1 - f_t) * c_{t-1}
            np.multiply(u, self.c[t-1], u)

            # Check if update is not for Wxf or bf
            if (X_OR_B == False):
                return u

            # Update common_sum used in update of dh/dWh
            # temp = f_t * (1 - f_t) * c_{t-1} * Whf
            temp = np.multiply(u, self.Whf)
            # common_sum += f_t * (1 - f_t) * c_{t-1} * Whf
            np.add(self.com_sum_dWh, temp, self.com_sum_dWh)

            # Check if update is for decoder
            if (self.Decoder):
                # temp = i_t * (1 - i_t) * c̃_t * Wxf[:, 0:self.WContext]
                temp = np.multiply(u, self.Wxf[:, 0:self.WContext])
                # Update common sum for partial w.r.t. context update
                np.add(self.com_sum_dContext, temp, self.com_sum_dContext)

        else:
            # u = f_t * (1 - f_t) * c_{t-1}
            np.multiply(u, self.cn1, u)

            # Check if update is for decoder
            if (self.Decoder):
                # temp = i_t * (1 - i_t) * c̃_t * Wxf[:, 0:self.WContext]
                temp = np.multiply(u, self.Wxf[:, 0:self.WContext])
                # Update common sum for partial w.r.t. context
                np.add(self.com_sum_dContext, temp, self.com_sum_dContext)

        # u = f_t * (1 - f_t) * c_{t-1} * o_t
        np.multiply(u, self.o[t], u)
        # v = tanh(c_t) * tanh(c_t)
        np.multiply(np.tanh(self.c[t]), np.tanh(self.c[t]), v)
        # v = (1 - (tanh(c_t))^2)
        np.subtract(np.ones(self.io_col), v, v)
        # u = f_t * (1 - f_t) * c_{t-1} * o_t * (1 - (tanh(c_t))^2)
        np.multiply(u, v, u)

        # Check if decoder model is in use
        if (self.Decoder):
            # Check if timestep > 0
            if (t > 0):
                # u = - dE_dh_t * f_t * (1 - f_t) * o_t * (1 - (tanh(c_t))^2) / N
                np.multiply(u, self.dE_dh[t], u)
                # Return array u
                return u
            else:    
                # Return array u
                return u
        # Otherwise encoder model in use
        else:
            return u

    # ------------------------- common_del_o ------------------------------ #
    # Returns the array which is the common multiple in dE_dWho, dE_dWxo, 
    # and dE_dbo given the following input:
    #   t : current timestep

    def common_del_o(self, t, X_OR_B):
        # Initialize temp arrays
        u = np.zeros(self.io_col)
        v = np.zeros(self.io_col)

        # v = (1 - o_t)
        np.subtract(np.ones(self.io_col), self.o[t], v)
        # u = (1 - o_t) * h_t
        np.multiply(self.h[t], v, u)

        # Check if decoder model is in use
        if (self.Decoder):
            # Check if timestep > 0
            if (t > 0):
                # Check if update is not for Wxf or bf
                if (X_OR_B == False):
                    return u

                # u = - dE_dh_t * (1 - o_t) * h_t / N_samples
                np.multiply(u, self.dE_dh[t], u)
                return u
            else:
                # Return array u
                return u
        # Otherwise encoder model in use
        else:
            return u

    # ------------------------- common_del_i ------------------------------ #
    # Returns the array which is the common multiple in dE_dWhi, dE_dWxi, 
    # and dE_dbi given the following input:
    #   t : current timestep
    #   X_OR_B : Boolean which is: True if update is for Wxf or bf
    #                              False otherwise

    def common_del_i(self, t, X_OR_B):
        # Initialize temp arrays
        u = np.zeros(self.io_col)
        v = np.zeros(self.io_col)

        # v = (1 - i_t)
        np.subtract(np.ones(self.io_col), self.i[t], v)
        # u = i_t * (1 - i_t)
        np.multiply(self.i[t], v, u)
        # u = i_t * (1 - i_t) * c̃_t
        np.multiply(u, self.c_tilde[t], u)

        # Check if update is for Wxf or bf
        if (X_OR_B):
            # Check if update is for decoder
            if (self.Decoder):
                # temp = i_t * (1 - i_t) * c̃_t * Wxi[:, 0:self.WContext]
                temp = np.multiply(u, self.Wxi[:, 0:self.WContext])
                # Update common sum for partial w.r.t. context
                np.add(self.com_sum_dContext, u, self.com_sum_dContext)

        # Check if timestep > 0
        if (t > 0):
            # Check if update is not for Wxf or bf
            if (X_OR_B == False):
                return u

            # Update common_sum used in update of dh/dWh
            # temp = i_t * (1 - i_t) * c̃_t * Whi
            temp = np.multiply(u, self.Whi)
            # common_sum += i_t * (1 - i_t) * c̃_t * Whi
            np.add(self.com_sum_dWh, temp, self.com_sum_dWh)

        # v = tanh(c_t) * tanh(c_t)
        np.multiply(np.tanh(self.c[t]), np.tanh(self.c[t]), v)
        # v = (1 - (tanh(c_t))^2)
        np.subtract(np.ones(self.io_col), v, v)
        # u = i_t * (1 - i_t) * c̃_t * (1 - (tanh(c_t))^2)
        np.multiply(u, v, u)
        # u = i_t * (1 - i_t) * c̃_t * (1 - (tanh(c_t))^2) * o_t
        np.multiply(u, self.o[t], u)


        # Check if decoder model is in use
        if (self.Decoder):
            # Check if update is requested for timestep > 0
            if (t > 0):
                # u = - dmu_dh_t * i_t * (1 - i_t) * c̃_t * (1 - (tanh(c_t))^2) * o_t / N
                np.multiply(u, self.dE_dh[t], u)
                # Return array u
                return u
            else:
                # Return array u
                return u    
        # Otherwise encoder model in use
        else:
            return u

    # ------------------------- common_del_c ------------------------------ #
    # Returns the array which is the common multiple in dE_dWhc, dE_dWxc, 
    # and dE_dbc given the following input:
    #   t : current timestep

    def common_del_c(self, t, X_OR_B):
        # Initialize temp arrays
        u = np.zeros(self.io_col)
        v = np.zeros(self.io_col)

        # v = c̃_t * c̃_t
        np.multiply(self.c_tilde[t], self.c_tilde[t], v)
        # v = (1 - (c̃_t)^2)
        np.subtract(np.ones(self.io_col), v, v)
        # u = i_t * (1 - (c̃_t)^2)
        np.multiply(self.i[t], v, u)

        # Check if update is for Wxf or bf
        if (X_OR_B):
            # Check if update is for decoder
            if (self.Decoder):
                # temp = i_t * (1 - (c̃_t)^2) * Wxc[:, 0:self.WContext]
                temp = np.multiply(u, self.Wxc[:, 0:self.WContext])
                # Update common sum for partial w.r.t. context
                np.add(self.com_sum_dContext, temp, self.com_sum_dContext)

        # Check if timestep > 0
        if (t > 0):
            # Check if update is not for Wxf or bf
            if (X_OR_B == False):
                return u

            # Update common_sum used in update of dh/dWh
            # temp = i_t * (1 - (c̃_t)^2) * Whc
            temp = np.multiply(u, self.Whc)
            # common_sum += i_t * (1 - (c̃_t)^2) * Whc
            np.add(self.com_sum_dWh, temp, self.com_sum_dWh)

        # Continue with update for common_del_c
        # u = o_t * i_t * (1 - (c̃_t)^2)
        np.multiply(u, self.o[t], u)
        # v = tanh(c_t) * tanh(c_t)
        np.multiply(np.tanh(self.c[t]), np.tanh(self.c[t]), v)
        # v = (1 - (tanh(c_t))^2)
        np.subtract(np.ones(self.io_col), v, v)
        # u = i_t * o_t * (1 - (c̃_t)^2) * (1 - (tanh(c_t))^2)
        np.multiply(u, v, u)

        # Check if decoder model is in use
        if (self.Decoder):
            # Check if update is requested for timestep > 0
            if (t > 0):
                # u = - dmu_dh_t * i_t * o_t * (1 - (c̃_t)^2) * (1 - (tanh(c_t))^2) / N
                np.multiply(u, self.dE_dh[t], u)
                # Return array u
                return u
            else:    # timestep == 0
                # Return array u
                return u    
        # Otherwise encoder model is in use
        else:
            return u


    # Routine to update the derivative values in the LSTM at a single timestep
    def density_der_update(self, t):
        # If t = 0 then we need to use different updates for dE_dWh weights
        if (t == 0):
            # ---------------- With respect to Forget Gate ---------------- #
            # Update common u
            u = self.common_del_f(t, True)

            # Set initial value for dh/dWhf
            temp = np.multiply(self.hn1, u)
            self.dh_dWhf = np.copy(temp)

            # Update dE/dWhf
            np.multiply(temp, self.dE_dh[0], temp)
            np.add(self.dE_dWhf, temp, self.dE_dWhf)

            # Update u
            np.multiply(u, self.dE_dh[0], u)

            # Update dE/dWxf
            temp = np.multiply(self.Ones_Wx, u[np.newaxis].T)
            temp = np.multiply(temp, self.x[t].T)
            np.add(self.dE_dWxf, temp, self.dE_dWxf)

            # Update dE/dbf
            np.add(self.dE_dbf, u, self.dE_dbf)

            # ---------------- With respect to Output Gate ---------------- #
            # Update common u
            u = self.common_del_o(t, True)

            # Set initial value for dh/dWho
            temp = np.multiply(self.hn1, u)
            self.dh_dWho = np.copy(temp)
        
            # Update dE/dWho
            np.multiply(temp, self.dE_dh[0], temp)
            np.add(self.dE_dWho, temp, self.dE_dWho)

            # Update u
            np.multiply(u, self.dE_dh[0], u)

            # Update dE/dWxo
            temp = np.multiply(self.Ones_Wx, u[np.newaxis].T)
            temp = np.multiply(temp, self.x[t].T)
            np.add(self.dE_dWxo, temp, self.dE_dWxo)

            # Update dE/dbo
            np.add(self.dE_dbo, u, self.dE_dbo)

            # ---------------- With respect to Input Gate ----------------- #
            # Update common u
            u = self.common_del_i(t, True)

            # Set initial value for dh/dWhi
            temp = np.multiply(self.hn1, u)
            self.dh_dWhi = np.copy(temp)

            # Update dE/dWhi
            np.multiply(temp, self.dE_dh[0], temp)
            np.add(self.dE_dWhi, temp, self.dE_dWhi)

            # Update u
            np.multiply(u, self.dE_dh[0], u)

            # Update dE/dWxi
            temp = np.multiply(self.Ones_Wx, u[np.newaxis].T)
            temp = np.multiply(temp, self.x[t].T)
            np.add(self.dE_dWxi, temp, self.dE_dWxi)

            # Update dE/dbi
            np.add(self.dE_dbi, u, self.dE_dbi)

            # ----------------- With respect to Cell Gate ----------------- #
            # Update common u
            u = self.common_del_c(t, True)

            # Set initial value for dh/dWhc
            temp = np.multiply(self.hn1, u)
            self.dh_dWhc = np.copy(temp)

            # Update dE/dWhc
            np.multiply(temp, self.dE_dh[0], temp)
            np.add(self.dE_dWhc, temp, self.dE_dWhc)

            # Update u
            np.multiply(u, self.dE_dh[0], u)

            # Update dE/dWxc
            temp = np.multiply(self.Ones_Wx, u[np.newaxis].T)
            temp = np.multiply(temp, self.x[t].T)
            np.add(self.dE_dWxc, temp, self.dE_dWxc)

            # Update dE/dbc
            np.add(self.dE_dbc, u, self.dE_dbc)

        else:
            # ---------------- With respect to Forget Gate ---------------- #
            # Update common u
            u = self.common_del_f(t, True)

            # Update dE/dWxf
            temp = np.multiply(self.Ones_Wx, u[np.newaxis].T)
            temp = np.multiply(temp, self.x[t].T)
            np.add(self.dE_dWxf, temp, self.dE_dWxf)

            # Update dE/dbf
            np.add(self.dE_dbf, u, self.dE_dbf)

            # ---------------- With respect to Output Gate ---------------- #
            # Update common u
            u = self.common_del_o(t, True)

            # Update dE/dWxo
            temp = np.multiply(self.Ones_Wx, u[np.newaxis].T)
            temp = np.multiply(temp, self.x[t].T)
            np.add(self.dE_dWxo, temp, self.dE_dWxo)

            # Update dE/dbo
            np.add(self.dE_dbo, u, self.dE_dbo)

            # ---------------- With respect to Input Gate ----------------- #
            # Update common u
            u = self.common_del_i(t, True)

            # Update dE/dWxi
            temp = np.multiply(self.Ones_Wx, u[np.newaxis].T)
            temp = np.multiply(temp, self.x[t].T)
            np.add(self.dE_dWxi, temp, self.dE_dWxi)

            # Update dE/dbi
            np.add(self.dE_dbi, u, self.dE_dbi)

            # ----------------- With respect to Cell Gate ----------------- #
            # Update common u
            u = self.common_del_c(t, True)

            # Update dE/dWxc
            temp = np.multiply(self.Ones_Wx, u[np.newaxis].T)
            temp = np.multiply(temp, self.x[t].T)
            np.add(self.dE_dWxc, temp, self.dE_dWxc)

            # Update dE/dbc
            np.add(self.dE_dbc, u, self.dE_dbc)

    # Routine to update the derivative values in the encoder at a single timestep
    def encoder_der_update(self, t):
        # If t < n-1 then we only need to update com_sum_dWh which is done as 
        # follows
        if (t < (self.time - 1)):
            # ---------------- With respect to Forget Gate ---------------- #
            # Update common u
            u = self.common_del_f(t, True)

            # ---------------- With respect to Output Gate ---------------- #
            # Update common u
            u = self.common_del_o(t, True)

            # ---------------- With respect to Input Gate ----------------- #
            # Update common u
            u = self.common_del_i(t, True)

            # ----------------- With respect to Cell Gate ----------------- #
            # Update common u
            u = self.common_del_c(t, True)

        else:
            # ---------------- With respect to Forget Gate ---------------- #
            # Update common u
            u = self.common_del_f(t, True)

            # Update dE/dWxf
            temp = np.multiply(self.Ones_Wx, u[np.newaxis].T)
            temp = np.multiply(temp, self.x[t].T)
            np.add(self.dE_dWxf, temp, self.dE_dWxf)

            # Update dE/dbf
            np.add(self.dE_dbf, u, self.dE_dbf)

            # ---------------- With respect to Output Gate ---------------- #
            # Update common u
            u = self.common_del_o(t, True)

            # Update dE/dWxo
            temp = np.multiply(self.Ones_Wx, u[np.newaxis].T)
            temp = np.multiply(temp, self.x[t].T)
            np.add(self.dE_dWxo, temp, self.dE_dWxo)

            # Update dE/dbo
            np.add(self.dE_dbo, u, self.dE_dbo)

            # ---------------- With respect to Input Gate ----------------- #
            # Update common u
            u = self.common_del_i(t, True)

            # Update dE/dWxi
            temp = np.multiply(self.Ones_Wx, u[np.newaxis].T)
            temp = np.multiply(temp, self.x[t].T)
            np.add(self.dE_dWxi, temp, self.dE_dWxi)

            # Update dE/dbi
            np.add(self.dE_dbi, u, self.dE_dbi)

            # ----------------- With respect to Cell Gate ----------------- #
            # Update common u
            u = self.common_del_c(t, True)

            # Update dE/dWxc
            temp = np.multiply(self.Ones_Wx, u[np.newaxis].T)
            temp = np.multiply(temp, self.x[t].T)
            np.add(self.dE_dWxc, temp, self.dE_dWxc)

            # Update dE/dbc
            np.add(self.dE_dbc, u, self.dE_dbc)


    # ------------------------ dh_dWh_update ----------------------------- #
    # Returns the array which is the common multiple in dE_dWxf and dE_dbf 
    # given the following input:
    #   t : current timestep

    def dh_dWh_update(self, t):
        # Initialize common arrays
        common1 = np.zeros(self.io_col)
        common2 = np.zeros(self.io_col)

        # Initialize temp arrays
        u = np.zeros((self.io_col, self.io_col))
        v = np.zeros((self.io_col, self.io_col))
        w = np.zeros((self.io_col, self.io_col))

        
        # ----------------------- Update commons -----------------------
        # Update common1
        # common1 = (1 - o_t) * h_t
        common1 = self.common_del_o(t, False)

        # Update common2
        # common2 = tanh(c_t) * tanh(c_t)
        np.multiply(np.tanh(self.c[t]), np.tanh(self.c[t]), common2)
        # common2 = 1 - (tanh(c[t]))^2
        np.subtract(np.ones(self.io_col), common2, common2)
        # common2 = o_t * (1 - (tanh(c[t]))^2)
        np.multiply(common2, self.o[t], common2)


        # ----------------------- Update dh/dWho -----------------------
        # w = Who * dh_{t-1}/dWho
        np.multiply(self.Who, self.dh_dWho, w)
        # w = h_{t-1} + Who * dh_{t-1}/dWho
        np.add(w, self.h[t-1], w)
        # w = (1 - o_t) * h_t * (h_{t-1} + Who * dh_{t-1}/dWho)
        np.multiply(w, common1, w)

        # v = common_sum * dh/dWho
        np.multiply(self.com_sum_dWh, self.dh_dWho, v)
        # u = common_sum * dh/dWho + dE_dWho_sum
        np.add(v, self.sum_dE_dWho, u)         
        # u = o_t * (1 - (tanh(c[t]))^2) * [common_sum * dh/dWho 
        #     + dE_dWho_sum]
        np.multiply(u, common2, u)
        # u = (1 - o_t) * h_t * (h_{t-1} + Who * dh/dWho) + o_t * 
        #     (1 - (tanh(c[t]))^2) * [common_sum * dh/dWho + dE_dWho_sum]
        np.add(u, w, u)

        # Update dh_t/dWho
        self.dh_dWho = np.copy(u)

        # Check if timestep < maximum number of timesteps
        if (t < (self.time - 1)):
            # Update sum_dE_dWho
            # v = f_{t+1} * common_sum * dh/dWho
            np.multiply(v, self.f[t+1], v)
            # sum_dE_dWho += f_{t+1} * common_sum * dh/dWho
            np.add(self.sum_dE_dWho, v, self.sum_dE_dWho)


        # ----------------------- Update dh/dContext -----------------------
        # Check if update for decoder model
        if (self.Decoder):
            vC = np.copy(self.com_sum_dContext)
            wC = np.zeros((self.Wrow, self.WContext))
            
            # wC = (1 - o_t) * h_t * Wxo[:, 0:self.WContext]
            np.multiply(common1, self.Wxo[:, 0:self.WContext], wC)

            # Update dc/dContext
            # Check if timestep > 0
            if (t > 0):
                # vC = common_sum + sum_dh_dContext
                np.add(self.com_sum_dContext, self.sum_dh_dContext, vC)

            # vC = o_t (1 - (tanh(c_t))^2) * (common_sum + sum_dh_dContext)
            np.multiply(vC, common2, vC)
            # vC = (1 - o_t) * h_t * Wxo[:, 0:self.WContext]
            #      + o_t (1 - (tanh(c_t))^2) * (common_sum + sum_dh_dContext)
            np.add(vC, wC, vC)

            # Update dh/dWhc
            self.dh_dContext = np.copy(vC)

            # Check if timestep < maximum number of timesteps
            if (t < (self.time - 1)):
                # Update sum_dh_dContext
                # vC = f_{t+1} * common_sum
                np.multiply(self.com_sum_dContext, self.f[t+1], vC)
                # sum_dh_dContext += f_{t+1} * common_sum
                np.add(self.sum_dh_dContext, vC, self.sum_dh_dContext)


        # ----------------------- Update common1 -----------------------
        # Update common1 before updating the rest of the partials
        # common1 = (1 - o_t) * h_t * Who
        np.multiply(common1, self.Who, w)

        # ----------------------- Update dh/dWhf -----------------------
        # w = (1 - o_t) * h_t * Who * dh/dWhf
        np.multiply(common1, self.dh_dWhf, w)

        # u1 = c_{t-1} * f_t * (1 - f_t)
        u1 = self.common_del_f(t, False)
        # u1 = c_{t-1} * f_t * (1 - f_t) * h_{t-1}
        np.multiply(u1, self.h[t-1], u1)
        # u = c_{t-1} * f_t * (1 - f_t) * h_{t-1} * E    (E = matrix of ones)
        np.multiply(u1, np.ones((self.io_col, self.io_col)), u)
        # v = common_sum * dh/dWhf
        np.multiply(self.com_sum_dWh, self.dh_dWhf, v)
        # u = c_{t-1} * f_t * (1 - f_t) * h_{t-1} * E + common_sum * dh/dWhf
        np.add(u, v, u)

        # Check if timestep > 0
        if (t > 0):
            # u = c_{t-1} * f_t * (1 - f_t) * h_{t-1} * E + common_sum * dh/dWhf
            #     + dE_dWhf_sum
            np.add(u, self.sum_dE_dWhf, u)
            
        # u = o_t * (1 - (tanh(c[t]))^2) * [c_{t-1} * f_t * (1 - f_t) * 
        #     h_{t-1} + common_sum * dh/dWhf + + dE_dWhf_sum]
        np.multiply(u, common2, u)

        # u = (1 - o_t) * h_t * Who * dh/dWhf + o_t * (1 - (tanh(c[t]))^2) * 
        #     [c_{t-1} * f_t * (1 - f_t) * h_{t-1} + common_sum * dh/dWhf 
        #     + dE_dWhf_sum]
        np.add(u, w, u)

        # Update dh/dWhf
        self.dh_dWhf = np.copy(u)

        # Check if timestep < maximum number of timesteps
        if (t < (self.time - 1)):
            # Update sum_dE_dWhf
            # v = f_{t+1} * common_sum * dh/dWhf
            np.multiply(v, self.f[t+1], v)
            # sum_dE_dWhf += f_{t+1} * common_sum * dh/dWhf
            np.add(self.sum_dE_dWhf, v, self.sum_dE_dWhf)


        # ----------------------- Update dh/dWhi -----------------------
        # w = (1 - o_t) * h_t * Who * dh/dWhi
        np.multiply(common1, self.dh_dWhi, w)

        # u1 = c̃_t * i_t * (1 - i_t)
        u1 = self.common_del_i(t, False)
        # u1 = c̃_t * i_t * (1 - i_t) * h_{t-1}
        np.multiply(u1, self.h[t-1], u1)
        # u = c̃_t * i_t * (1 - i_t) * h_{t-1} * E    (E = matrix of ones)
        np.multiply(u1, np.ones((self.io_col, self.io_col)), u)
        # v = common_sum * dh/dWhi
        np.multiply(self.com_sum_dWh, self.dh_dWhi, v)
        # u = c̃_t * i_t * (1 - i_t) * h_{t-1} + common_sum * dh/dWhi
        np.add(u, v, u)

        # Check if timestep > 0
        if (t > 0):
            # u = c̃_t * i_t * (1 - i_t) * h_{t-1} + common_sum * dh/dWhi
            #     + dE_dWhi_sum
            np.add(u, self.sum_dE_dWhi, u)
            
        # u = o_t * (1 - (tanh(c[t]))^2) * [c̃_t * i_t * (1 - i_t) * 
        #     h_{t-1} + common_sum * dh/dWhi + dE_dWhi_sum]
        np.multiply(u, common2, u)
        # u = (1 - o_t) * h_t * Who * dh/dWhi + o_t * (1 - (tanh(c[t]))^2) * 
        #     [c̃_t * i_t * (1 - i_t) * h_{t-1} + common_sum * dh/dWhi 
        #     + dE_dWhi_sum]
        np.add(u, w, u)

        # Update dh/dWhi
        self.dh_dWhi = np.copy(u)

        # Check if timestep < maximum number of timesteps
        if (t < (self.time - 1)):
            # Update sum_dE_dWhi
            # v = f_{t+1} * common_sum * dh/dWhi
            np.multiply(v, self.f[t+1], v)
            # sum_dE_dWhi += f_{t+1} * common_sum * dh/dWhi
            np.add(self.sum_dE_dWhi, v, self.sum_dE_dWhi)


        # ----------------------- Update dh/dWhc -----------------------
        # w = (1 - o_t) * h_t * Who * dh/dWhi
        np.multiply(common1, self.dh_dWhi, w)

        # u1 = i_t * (1 - (c̃_t)^2)
        u1 = self.common_del_c(t, False)
        # u1 = i_t * (1 - (c̃_t)^2) * h_{t-1}
        np.multiply(u1, self.h[t-1], u1)
        # u1 = i_t * (1 - (c̃_t)^2) * h_{t-1} * E    (E = matrix of ones)
        np.multiply(u1, np.ones((self.io_col, self.io_col)), u)
        # v = common_sum * dh/dWhc
        np.multiply(self.com_sum_dWh, self.dh_dWhc, v)
        # u = i_t * (1 - (c̃_t)^2) * h_{t-1} + common_sum * dh/dWhi
        np.add(u, v, u)

        # Check if timestep > 0
        if (t > 0):
            # u = c̃_t * i_t * (1 - i_t) * h_{t-1} + common_sum * dh/dWhi
            #     + dE_dWhi_sum
            np.add(u, self.sum_dE_dWhc, u)
            
        # u = o_t * (1 - (tanh(c[t]))^2) * [i_t * (1 - (c̃_t)^2) * h_{t-1}
        #     + common_sum * dh/dWhi + dE_dWhi_sum]
        np.multiply(u, common2, u)
        # u = (1 - o_t) * h_t * Who * dh/dWhi + o_t * (1 - (tanh(c[t]))^2) * 
        #     [i_t * (1 - (c̃_t)^2) * h_{t-1} + common_sum * dh/dWhi 
        #     + dE_dWhi_sum]
        np.add(u, w, u)

        # Update dh/dWhc
        self.dh_dWhc = np.copy(u)

        # Check if timestep < maximum number of timesteps
        if (t < (self.time - 1)):
            # Update sum_dE_dWhf
            # v = f_{t+1} * common_sum * dh/dWhc
            np.multiply(v, self.f[t+1], v)
            # sum_dE_dWhc += f_{t+1} * common_sum * dh/dWhc
            np.add(self.sum_dE_dWhc, v, self.sum_dE_dWhc)


    def density_der_update_h(self, t):
        # Initialize temp arrays
        u = np.zeros((self.io_col, self.io_col))

        # -------------------- Update com_sum_dWh --------------------- #
        # In order to save on floating point operations, the update for
        # com_sum_dWh has been coded into the routine der_update as 
        # part of the routines common_del_f, common_del_i, and
        # common_del_c.

        # ---------------- Update dh/dWh for all gates ---------------- #
        self.dh_dWh_update(t)

        # ---------------- With respect to Forget Gate ---------------- #          
        # Update dE/dWhf
        np.multiply(self.dh_dWhf, self.dE_dh[t], u)
        np.add(self.dE_dWhf, u, self.dE_dWhf)

        # ---------------- With respect to Output Gate ---------------- #
        # Update dE/dWho
        np.multiply(self.dh_dWho, self.dE_dh[t], u)
        np.add(self.dE_dWho, u, self.dE_dWho)

        # ---------------- With respect to Input Gate ----------------- #
        # Update dE/dWhi
        np.multiply(self.dh_dWhi, self.dE_dh[t], u)
        np.add(self.dE_dWhi, u, self.dE_dWhi)

        # ----------------- With respect to Cell Gate ----------------- #
        # Update dE/dWhc
        np.multiply(self.dh_dWhc, self.dE_dh[t], u)
        np.add(self.dE_dWhc, u, self.dE_dWhc)

        # ------------------ With respect to Context ------------------ #
        # Update dE/dContext
        np.multiply(self.dh_dContext, self.dE_dh[t], u)
        np.add(self.dE_dContext, u, self.dE_dContext)


    def encoder_der_update_h(self, t):
        # Initialize temp arrays
        u = np.zeros((self.io_col, self.io_col))

        # -------------------- Update com_sum_dWh --------------------- #
        # In order to save on floating point operations, the update for
        # com_sum_dWh has been coded into the routine der_update as 
        # part of the routines common_del_f, common_del_i, and
        # common_del_c.

        # ---------------- Update dh/dWh for all gates ---------------- #
        self.dh_dWh_update(t)

        # ---------------- With respect to Forget Gate ---------------- #          
        # Update dE/dWhf
        np.multiply(self.dh_dWhf, self.dE_dDecoder[t], u)
        np.add(self.dE_dWhf, u, self.dE_dWhf)

        # ---------------- With respect to Output Gate ---------------- #
        # Update dE/dWho
        np.multiply(self.dh_dWho, self.dE_dDecoder[t], u)
        np.add(self.dE_dWho, u, self.dE_dWho)

        # ---------------- With respect to Input Gate ----------------- #
        # Update dE/dWhi
        np.multiply(self.dh_dWhi, self.dE_dDecoder[t], u)
        np.add(self.dE_dWhi, u, self.dE_dWhi)

        # ----------------- With respect to Cell Gate ----------------- #
        # Update dE/dWhc
        np.multiply(self.dh_dWhc, self.dE_dDecoder[t], u)
        np.add(self.dE_dWhc, u, self.dE_dWhc)


    # Routine to update search directions for stochastic gradient descent
    def direction_update(self):
        # ---------------- Update Forget Gate Weights ---------------- #
        # Hidden weights
        if (np.isfinite(self.dE_dWhf).all()):
            np.add(self.dir.Whf, self.dE_dWhf, self.dir.Whf)

        # Input weights
        if (np.isfinite(self.dE_dWxf).all()):
            np.add(self.dir.Wxf, self.dE_dWxf, self.dir.Wxf)

        # Bias terms
        if (np.isfinite(self.dE_dbf).all()):
            np.add(self.dir.bf, self.dE_dbf, self.dir.bf)

        # ---------------- Update Output Gate Weights ---------------- #
        # Hidden weights
        if (np.isfinite(self.dE_dWho).all()):
            np.add(self.dir.Who, self.dE_dWho, self.dir.Who)

        # Input weights
        if (np.isfinite(self.dE_dWxo).all()):
            np.add(self.dir.Wxo, self.dE_dWxo, self.dir.Wxo)

        # Bias terms
        if (np.isfinite(self.dE_dbo).all()):
            np.add(self.dir.bo, self.dE_dbo, self.dir.bo)

        # ---------------- Update Input Gate Weights ----------------- #
        # Hidden weights
        if (np.isfinite(self.dE_dWhi).all()):
            np.add(self.dir.Whi, self.dE_dWhi, self.dir.Whi)

        # Input weights
        if (np.isfinite(self.dE_dWxi).all()):
            np.add(self.dir.Wxi, self.dE_dWxi, self.dir.Wxi)

        # Bias terms
        if (np.isfinite(self.dE_dbi).all()):
            np.add(self.dir.bi, self.dE_dbi, self.dir.bi)

        # ----------------- Update Cell Gate Weights ----------------- #
        # Hidden weights
        if (np.isfinite(self.dE_dWhc).all()):
            np.add(self.dir.Whc, self.dE_dWhc, self.dir.Whc)

        # Input weights
        if (np.isfinite(self.dE_dWxc).all()):
            np.add(self.dir.Wxc, self.dE_dWxc, self.dir.Wxc)

        # Bias terms
        if (np.isfinite(self.dE_dbc).all()):
            np.add(self.dir.bc, self.dE_dbc, self.dir.bc)

        # Check if decoder model is in use
        if (self.Decoder):
            if (np.isfinite(self.dE_db_mu).all()):
                np.add(self.dir.b_mu, self.dE_db_mu, self.dir.b_mu)
      


    # Routine for updating weights and biases using derivatives
    def weight_update(self):
        # Before updating each weight, we check to see if the updates 
        # are fininte (i.e. none of the updates are np.inf, -np.inf, or
        # np.nan)
        # ---------------- Update Forget Gate Weights ---------------- #
        # Hidden weights
        step = self.lr * self.dir.Whf
        np.subtract(self.Whf, step, self.Whf)

        # Input weights
        step = self.lr * self.dir.Wxf
        np.subtract(self.Wxf, step, self.Wxf)

        # Bias terms
        step = self.lr * self.dir.bf
        np.subtract(self.bf, step, self.bf)

        # ---------------- Update Output Gate Weights ---------------- #
        # Hidden weights
        step = self.lr * self.dir.Who
        np.subtract(self.Who, step, self.Who)

        # Input weights
        step = self.lr * self.dir.Wxo
        np.subtract(self.Wxo, step, self.Wxo)

        # Bias terms
        step = self.lr * self.dir.bo
        np.subtract(self.bo, step, self.bo)

        # ---------------- Update Input Gate Weights ----------------- #
        # Hidden weights
        step = self.lr * self.dir.Whi
        np.subtract(self.Whi, step, self.Whi)

        # Input weights
        step = self.lr * self.dir.Wxi
        np.subtract(self.Wxi, step, self.Wxi)

        # Bias terms
        step = self.lr * self.dir.bi
        np.subtract(self.bi, step, self.bi)

        # ----------------- Update Cell Gate Weights ----------------- #
        # Hidden weights
        step = self.lr * self.dir.Whc
        np.subtract(self.Whc, step, self.Whc)

        # Input weights
        step = self.lr * self.dir.Wxc
        np.subtract(self.Wxc, step, self.Wxc)

        # Bias terms
        step = self.lr * self.dir.bc
        np.subtract(self.bc, step, self.bc)

        # Check if decoder model is in use
        if (self.Decoder):
            step = self.lr * self.dir.b_mu
            np.subtract(self.b_mu, step, self.b_mu)

        # Reset update directions
        self.dir.reset_directions()


class Encoder:
    # Initialize elements in class
    def __init__(self, n_x, x_dim, n_io, n_time, learning_rate,
                 dict_filename, Save_Init_Weights, Testing, USER_INPUT):

        # Initialize parameters (states, weights, and bias terms)
        self.p = Parm(n_x, x_dim, n_io, n_time, learning_rate, 
                      dict_filename, 1, False, Save_Init_Weights,
                      Testing, USER_INPUT)


    def forward(self):
        # Set states to have the number of timesteps corresponding
        # to the current sample
        self.p.states_reset()

        for i in range(self.p.time):
            # Update all states
            self.p.cell_state_update(i)

        
    def backward(self):
        # Reset all partials to 0 before updating
        self.p.der_reset()
        # Update all partials at timestep = 0
        self.p.encoder_der_update(0)

        for i in range(1, self.p.time):
            # Reset common sum for dh/dWh before updating
            self.p.der_reset_common_sum()
            # Update all partials
            self.p.encoder_der_update(i)
            self.p.dh_dWh_update(i)
            self.p.encoder_der_update_h(i)

        self.p.direction_update()
        
    def gradient_descent(self):
        self.p.weight_update()

class Decoder:
    # Initialize elements in class
    def __init__(self, n_x, x_dim, n_io, n_time, learning_rate, 
                 dict_filename, num_samples, Save_Init_Weights, Testing,
                 USER_INPUT):

        # Initialize parameters (states, weights, and bias terms)
        self.p = Parm(n_x, x_dim, n_io, n_time, learning_rate, 
                      dict_filename, num_samples, True, Save_Init_Weights,
                      Testing, USER_INPUT)


    def train_forward(self):
        # Set states to have the number of timesteps corresponding
        # to the current sample
        self.p.states_reset()

        # Reset the dE_dh and dE_db_mu that are computed inside the softmax
        self.p.dE_dh_reset()

        for i in range(self.p.time):
            # Update all states
            self.p.cell_state_update(i)
            self.p.mu_update(i)
            self.p.density_update(i)

        # Update total error
        self.p.total_density_update()


    def test_forward(self):
        # Set states to have the number of timesteps corresponding
        # to the current sample
        self.p.states_reset()

        for i in range(self.p.time):
            # Check if user input is being used
            if (self.p.USER_INPUT):
                # Update columns 300 - 600 of input x[i]
                self.p.test_x_update(i)

            # Update all states
            self.p.cell_state_update(i)
            self.p.mu_update(i)

            if (self.p.USER_INPUT == True):
                if (self.p.current_token == '</eos>'):
                    break

        
    def backward(self):
        # Reset all partials to 0 before updating
        self.p.der_reset()

        # Update all partial values at timestep = 0
        self.p.density_der_update(0)

        # Update all partial values for timestep = 1 to end of sentence
        for i in range(1, self.p.time):
            # Reset common sum for dh/dWh before updating
            self.p.der_reset_common_sum()
            # Update all partials
            self.p.density_der_update(i)
            self.p.dh_dWh_update(i)
            self.p.density_der_update_h(i)

        self.p.direction_update()
        
    def gradient_descent(self):
        self.p.weight_update()

class Enc_Dec:
    # Initialize encoder and decoder
    def __init__(self, x_dim, n_io, n_time, learning_rate, 
                 source_dict_filename, target_dict_filename, 
                 num_samples, Save_Init_Weights, Testing, USER_INPUT):
        self.x_dim = x_dim      

        # Initialize encoder
        self.Enc = Encoder(1, x_dim[0], n_io, n_time, learning_rate, 
                           source_dict_filename, Save_Init_Weights, Testing,
                           USER_INPUT)

        # Initialize decoder
        self.Dec = Decoder(2, x_dim, n_io, n_time, learning_rate, 
                           target_dict_filename, num_samples,
                           Save_Init_Weights, Testing, USER_INPUT)


    # --------------------- Forward Propagation --------------------- # 
    # ------------------------- For Training ------------------------ # 
    def train_forward(self, source_filename, target_filename, sample_index):
        # Update input, token array, and number of timesteps for encoder model
        self.Enc.p.input_update(source_filename, sample_index, 0)

        # Forward propagation for encoder
        self.Enc.forward()

        # Update input, token array, and number of timesteps for decoder model
        self.Dec.p.input_update(target_filename, sample_index, 
                                self.Enc.p.h[self.Enc.p.time-1])

        # Forward propagation for decoder
        self.Dec.train_forward()

        # Print the output densities
#        print("Densities = ", self.Dec.p.density)

        print("Total Loss for sample = ", self.Dec.p.total_density)

    # ------------------------- For Testing ------------------------- # 
    def test_forward(self, source_filename, target_filename, sample_index):
        # Update input, token array, and number of timesteps for encoder model
        self.Enc.p.input_update(source_filename, sample_index, 0)

        # Forward propagation for encoder
        self.Enc.forward()

        # Update input, token array, and number of timesteps for decoder model
        self.Dec.p.input_update(target_filename, sample_index, 
                                self.Enc.p.h[self.Enc.p.time-1])

        # Forward propagation for decoder
        self.Dec.test_forward()

    # -------------------- Backward Propagation --------------------- # 
    def backward(self):
        # Backward propagation for decoder
        self.Dec.backward()

        # Update partial derivative to be passed from decoder back to encoder
        self.Enc.p.decoder_to_encoder_partial_update(self.Dec.p.dE_dContext)

        # Backward propagation for encoder
        self.Enc.backward()

    # ----------------- Stochastic Gradient Descent ----------------- # 
    def stochastic_gradient(self, source_filename, target_filename, batch):
        for i in batch:
            # Perform forward propagation
            self.train_forward(source_filename, target_filename, i)

            # Perform backward propagation
            self.backward()

        # Perform gradient descent for decoder
        self.Dec.gradient_descent()
        # Perform gradient descent for encoder
        self.Enc.gradient_descent()

    # -------------------- Update Initial Weights ------------------- # 
    # This routine is to store the current weights so that they can 
    # be loaded for future training
    def update_initial_weights(self):
        # Save weights for encoder model
        self.Enc.p.save_weights_to_file(False)

        # Save weights for decoder model
        self.Dec.p.save_weights_to_file(False)

    # ---------------------- Save Final Weights --------------------- # 
    def save_final_weights(self):
        # Save weights for encoder model
        self.Enc.p.save_weights_to_file(True)

        # Save weights for decoder model
        self.Dec.p.save_weights_to_file(True)



def train(max_epochs, Save_Init_Weights, Save_Final_Weights):
    # Initialize encoder-decoder class
    M = Enc_Dec(np.array([300, 300]),   # x_dim
                300,                    # n_io
                4,                      # n_time
                1,                      # learning_rate
                'en.vec',               # source_dict_filename
                'fr.vec',               # target_dict_filename
                1,                     # batch_size
                Save_Init_Weights,      # Save_Init_Weights
                False,                  # Testing
                False)                  # User is not inputting sentence

    # Initialize filenames for source and target 
    source_filename = 'europarl-v7.fr-en.en'
    target_filename = 'europarl-v7.fr-en.fr'

    # Initialize sample_index
    sample_index = 1 #420 #45

    # Initialize end time
    t_end = time.time() + 60 * 60 * 20

    # Initialize number of epochs
    n_epochs = 0

    # Initialize batch size
    batch_size = 10

    for i in range(max_epochs):
#    while time.time() < t_end:
        # Choose new sample_index
#        batch = np.random.randint(1, 2007000, batch_size)  
        batch = np.array([sample_index])

        # Perform forward propagation on one sample
#        M.train_forward(source_filename, target_filename, batch)

        # Perform backward propagation on one sample
#        M.backward()

        # Perform stochastic gradient descent
        M.stochastic_gradient(source_filename, target_filename, batch)

        # Print number of epochs
        n_epochs += 1
        print("---------------------------------------")
        print(" n_epochs = ", n_epochs)
        print("---------------------------------------")

        #M.update_initial_weights()
        M.save_final_weights()

    # Perform additional forward propagation to check if error has reduced
#    M.train_forward(source_filename, target_filename, sample_index)

    # Save final weights to file for testing
    # Check if final weights are to be saved for testing
#    if (Save_Final_Weights):
    M.save_final_weights()


def auto_test():
    # Initialize encoder-decoder class
    M = Enc_Dec(np.array([300, 300]),   # x_dim
                300,                    # n_io
                6,                      # maximum n_time
                1,                      # learning_rate
                'en.vec',               # source_dict_filename
                'fr.vec',               # target_dict_filename
                10,                     # num_samples
                False,                  # Save_Init_Weights
                True,                   # Testing
                False)                  # User is not inputting sentence

    # Initialize filenames for source and target 
    source_filename = 'europarl-v7.fr-en.en'
    target_filename = 'europarl-v7.fr-en.fr'

    # Initialize sample_index
    sample_index = 1 #420 #45

    for i in range(1, 2):
        # Perform forward propagation on one sample
        M.test_forward(source_filename, target_filename, i) #sample_index)

        # End line
        print("", end='\n')


# Automated train and test
def train_and_auto_test():
    train(100, False, True)

    auto_test()


# User input test
def user_test(max_out_length):
    # Initialize encoder-decoder class
    M = Enc_Dec(np.array([300, 300]),   # x_dim
                300,                    # n_io
                max_out_length,         # maximum n_time
                1,                      # learning_rate
                'en.vec',               # source_dict_filename
                'fr.vec',               # target_dict_filename
                1,                      # num_samples
                False,                  # Save_Init_Weights
                True,                   # Testing
                True)                   # User is inputting sentence

    # Initialize indicator that program should prompt for translation
    translate = 'y'

    while (translate == 'y' or translate == 'Y'):
        # Prompt user to input source sentence for translation
        source_sentence = input("\n Enter the English sentence you would like "
                            "to translate to French:\n ")
        print("\n Source sentence:\n " + source_sentence)

        # Perform the machine translation procedure and output translation
        print("\n Translated sentence:\n ", end="")

        M.test_forward(source_sentence, '', 0)

        print("", end='\n')

        # Prompt the user to indicate if they would like to perform 
        # another translation
        translate = input("\n Would you like to perform another translation? "
                          "(\'y\' for yes or \'n\' for no):\n ")


#train_and_auto_test()

user_test(10)
