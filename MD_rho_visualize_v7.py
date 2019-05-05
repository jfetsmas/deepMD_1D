import tensorflow as tf
from time import gmtime, strftime
import os
import timeit
import matplotlib.pyplot as plt
import h5py
import numpy as np
import random
np.random.seed(123)  # for reproducibility
#random.seed(123)
import math

# model path names
data_folder = 'data/'
log_folder = 'logs/'
models_folder = 'models/'  # folder to save the models (weights)
input_prefix = 'KS_MD_scf_8_sigma_6.0'
script__name = 'MD_rho_1D_v7_py'
alpha = 64
n_train = 20
n_test = 5
intLength = 80.0
N_near = 7

# this is generic for al the test_files
dir__name =  os.getcwd()
print("local directoty")
print(dir__name)




filenameIpt = data_folder + 'Input_'  + input_prefix + '.h5'
filenameOpt = data_folder + 'Output_' + input_prefix + '.h5'

# import data: size of data: Nsamples * Nx
fInput = h5py.File(filenameIpt,'r')
InputArray = fInput['R'][:].T

fOutput = h5py.File(filenameOpt,'r')
RhoT = fOutput['Rho'][:]
Etotal = fOutput['Etotal'][:]
Ftotal = fOutput['Ftotal'][:]

Nsamples = InputArray.shape[0]
Nx = RhoT.shape[1]


### some pre-processing of the date
Etot_mean = np.mean(Etotal)
Etot_sigma = np.std(Etotal)

Etotal = (Etotal-Etot_mean)

n_input = InputArray.shape[1]
Natoms = n_input

Sample_picked = 2700

# building the grid for the x data to be fed
Xpos = (np.array(range(Nx))*intLength/float(Nx))

# Input has xpos, Rpos
Input2 = np.array( [ [ np.concatenate(([r],InputArray[Sample_picked,:])) ] for r in Xpos])
# Output has Energy, Forces and Rho
Output2 = np.array( [np.concatenate((Etotal[Sample_picked], Ftotal[Sample_picked,:], [r]))  for r in RhoT[Sample_picked,:]] )

# preraring the data both input and output
InputArray = Input2.reshape(Nx, Natoms+1)
OutputArray = Output2.reshape(Nx,Natoms+2)

# modifying the definition of the number of samples
Nsamples = Nsamples*Nx
n_train = n_train*Nx
n_test = n_test*Nx

Xpos_picked = Xpos
RhoT_picked = OutputArray[:,-1]

xtest  = InputArray
ytest  = OutputArray






#------------------------------------------------------------------------------#
# Writting the model using tensoflow
class denseLayerPyramid:

  def __init__(self, Input_layer, alpha, 
               actfn = tf.nn.tanh, atomType='C', 
               scope = "Dense", reuse=None):
    
    with tf.variable_scope(scope, reuse=reuse):
    # writting only a few hidden layers (why not right?)
      self._hidden  = tf.layers.dense(Input_layer, alpha, actfn, name='dl_'+atomType+'_1')
      self._hidden1 = tf.layers.dense(self._hidden, alpha//2, actfn, name='dl_'+atomType+'_2')
      self._hidden2 = tf.layers.dense(self._hidden1, alpha//4, actfn, name='dl_'+atomType+'_3')
      self._hidden3 = tf.layers.dense(self._hidden2, alpha//8, actfn, name='dl_'+atomType+'_4')
      self._hidden4 = tf.layers.dense(self._hidden3, alpha//16, actfn, name='dl_'+atomType+'_5')
      self._hidden5 = tf.layers.dense(self._hidden4, alpha//32, actfn, name='dl_'+atomType+'_6')
      self._output = tf.layers.dense(self._hidden5, alpha//64, name='dl_'+atomType+'_7')

  def __call__(self):
    # we define a callable object       
    return(self._output)

def index_matrix_to_pairs(index_matrix):
  replicated_first_indices = tf.tile(
      tf.expand_dims(tf.range(tf.shape(index_matrix)[0]), dim=1), 
      [1, tf.shape(index_matrix)[1]])
  return tf.stack([replicated_first_indices, index_matrix], axis=2)

# we reset the graph (very useful in interactive mode)
tf.reset_default_graph()

X = tf.placeholder(tf.float64, shape=(None, 1), name='input_X')
R = tf.placeholder(tf.float64, shape=(None, Natoms), name='input_R')

Y = tf.placeholder(tf.float64, shape=(None, Natoms+2), name='output')

Y_Energy = Y[:,0:1]
Y_Forces = Y[:,1:Natoms+1]
Y_Rho = Y[:,Natoms+1:]

### We build the network for the Energy and Forces
R_Array = []
Diff_Array = []
Diff_Array_Per = []
Dist_Array= []
Div_Array = []
for i in range(Natoms):
  for j in range(Natoms):
    if i != j:
      # R_J - R_I
      Diff_Array.append(tf.expand_dims(tf.subtract(R[:,j], R[:,i]),-1))
      # periodic R_J - R_I
      Diff_Array_Per.append(tf.subtract(Diff_Array[-1], tf.scalar_mul(intLength, tf.round(tf.scalar_mul(1./(intLength), Diff_Array[-1])))))
      # ||  R_J - R_I  ||
      Dist_Array.append(tf.abs(Diff_Array_Per[-1]))
      #  R_J - R_I /||  R_J - R_I  ||
      Div_Array.append(tf.multiply(tf.reciprocal(Dist_Array[-1]), Diff_Array_Per[-1] ))
      # concatenating ||R_J - R_I|| with  R_J - R_I /||  R_J - R_I  ||
      R_Array.append(tf.concat([Dist_Array[-1], Div_Array[-1]], axis = 1))


R_IJ_H = []
for i in range(0,Natoms):
  R_pre = tf.reshape(tf.concat(R_Array[(Natoms-1)*i:(i+1)*(Natoms-1)], axis = 1),[-1,Natoms-1,2])
  Dist = tf.concat(Dist_Array[(Natoms-1)*i:(i+1)*(Natoms-1)], axis = 1)
  values, indices = tf.nn.top_k(tf.scalar_mul(-1.0,Dist),N_near)
  R_post = tf.gather_nd(R_pre, index_matrix_to_pairs(indices))
  R_IJ_H.append(tf.reshape(R_post, [-1,2*N_near]))

PH_Array = []
PH_Array_calculation = []
PH_1 = denseLayerPyramid(R_IJ_H[0], alpha, atomType='H', scope='Dense_H')
PH_Array.append(PH_1)
PH_Array_calculation.append(PH_1())
for i in range(1,Natoms):
  PH_i = denseLayerPyramid(R_IJ_H[i], alpha, atomType='H', scope='Dense_H', reuse=True)
  PH_Array_calculation.append(PH_i())
  PH_Array.append(PH_i)

final = tf.concat(PH_Array_calculation, axis =1)

# outputing the energy
Energy = tf.expand_dims(tf.reduce_sum(final, axis = 1), 1)

dEdR = tf.gradients(Energy, R, stop_gradients=[R])
Forces = tf.scalar_mul(-1. , tf.squeeze(dEdR, [0]))

RX = tf.expand_dims(tf.subtract(R,tf.tile(X, [1, Natoms])),-1)
# periodicing the distance
RXper = tf.subtract(RX, tf.scalar_mul(intLength, tf.round(tf.scalar_mul(1./(intLength), RX))))
DisTRX = tf.abs(RXper)

RAUG = tf.concat([ DisTRX, RXper ], axis = 2 )

H_input_Array = []
rho_H_Array = []
rho_H_exp_Array = []
H_input_1 = tf.concat( [RAUG[:,0,:],PH_1._hidden4], axis=1)
rho_H_1 = denseLayerPyramid(H_input_1, alpha, actfn = tf.nn.relu, 
                            atomType='H', scope='Dense_rho_H')
rho_H_1_exp = denseLayerPyramid(H_input_1, alpha, actfn = tf.nn.relu, 
                            atomType='H', scope='Dense_rho_H_exp')
H_input_Array.append(H_input_1)
rho_H_Array.append(rho_H_1)
rho_H_exp_Array.append(rho_H_1_exp)
for i in range(1,Natoms):
  H_input_i = tf.concat( [RAUG[:,i,:],PH_Array[i]._hidden4], axis=1)
  rho_H_i = denseLayerPyramid(H_input_i, alpha, actfn = tf.nn.relu, 
                            atomType='H', reuse=True, scope='Dense_rho_H')
  rho_H_i_exp = denseLayerPyramid(H_input_i, alpha, actfn = tf.nn.relu, 
                            atomType='H', reuse=True, scope='Dense_rho_H_exp')
  H_input_Array.append(H_input_i)
  rho_H_Array.append(rho_H_i)
  rho_H_exp_Array.append(rho_H_i_exp)

with tf.variable_scope("exp_H", reuse=True):
  # weight_init = np.ones((1, alpha//64))  + 0.001*(np.random.rand(1, alpha//64)-0.5)
  # bias_init = np.zeros((1, alpha//64))  + 0.001*(np.random.rand(1, alpha//64))

  weight_H = tf.Variable( 1., dtype = tf.float64, trainable = True, name = "weight_H")
  bias_H = tf.Variable(0., dtype = tf.float64, trainable = True, name = "bias_H")
  
  rho_exp_H_Array = []
  mult_H_Array = []

  for i in range(0,Natoms):
    mult_H_i = tf.scalar_mul(weight_H, RAUG[:,i,0:1] - bias_H)
    rho_exp_H_i = tf.multiply(tf.exp(-tf.abs(rho_H_exp_Array[i]._output + mult_H_i)), rho_H_Array[i]._output)
    mult_H_Array.append(mult_H_i)
    rho_exp_H_Array.append(rho_exp_H_i)

final_rho = tf.concat(rho_exp_H_Array, axis =1)

Rho = tf.expand_dims(tf.reduce_sum(final_rho, axis = 1), 1)


# defining the loss and cost functions
loss_E = tf.square(tf.subtract(Energy, Y_Energy))
loss_F = tf.reduce_sum(tf.square(tf.subtract(Forces, Y_Forces)), axis = 1)
loss_rho = tf.square(tf.subtract(Rho, Y_Rho))

loss = loss_E + 100000*loss_rho + 100*loss_F
cost = tf.reduce_mean(loss)

#------------------------------------------------------------------------------#





# initialize the graph and the variables within
init = tf.global_variables_initializer()

saver = tf.train.Saver()

modelsavefolder = models_folder + 'model_'+ script__name + \
                 '_Input_' + input_prefix +\
                 '_alpha_' + str(alpha) + \
                 '_n_train_' + str(n_train) + \
                 '_n_test_' + str(n_test) + \
                 '_Nnear_' + str(N_near)


with tf.Session() as sess:

  sess.run(tf.global_variables_initializer())

  saver.restore(sess, dir__name+"/"+modelsavefolder+"/model.ckpt")
  print("Best Model restored")
  loss_test = sess.run(cost, feed_dict= {X: xtest[:,0:1],R:xtest[:,1:], Y: ytest})
  print('Test Loss from best model: %.8e'  % loss_test)

  Rho_out = sess.run(Rho, feed_dict= { X: xtest[:,0:1], R: xtest[:,1:], Y: ytest})
  Rho_picked = np.reshape(RhoT_picked,(Nx,1))

  plt.plot(Xpos_picked, Rho_picked)
  plt.plot(Xpos_picked, Rho_out,'r--')
  # plt.plot(Xpos_picked, Rho_out-Rho_picked)
  plt.savefig(input_prefix+'_Nnear_'+str(N_near)+'.png', bbox_inches="tight")





# possible strategies
# normalize the energy (it is very big compared to the rest)
# learn the energy first along with rho, and at the end use taht to lear the forces