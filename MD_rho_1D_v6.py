import tensorflow as tf
from time import gmtime, strftime
import os
import timeit
import argparse
import h5py
import numpy as np
import random
#np.random.seed(123)  # for reproducibility
#random.seed(123)
import math

parser = argparse.ArgumentParser(description='Ab-initio MD')
parser.add_argument('--epoch', type=int, default=200, metavar='N',
                    help='input number of epochs for training (default: 200)')
parser.add_argument('--input-prefix', type=str, default='KS_MD_scf_3_sigma_0.75', metavar='N',
                    help='prefix of input data filename (default: KS)')
parser.add_argument('--alpha', type=int, default=256, metavar='N',
                    help='input number of channels for training (default: 64)')
parser.add_argument('--L', type=int, default=7, metavar='N',
                    help='input number of levels (default: 7)')
parser.add_argument('--n-cnn', type=int, default=6, metavar='N',
                    help='input number layer of CNNs (default: 5)')
parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',
                    help='learning rate (default: 1e-3)')
parser.add_argument('--batch-size', type=int, default=0, metavar='N',
                    help='batch size (default: 1/100 of the training )')
parser.add_argument('--inter-size', type=int, default=7, metavar='N',
                    help='number of points in interpolation(default: 7)')
parser.add_argument('--restr-size', type=int, default=3, metavar='N',
                    help='number of points in restriction(default: 3)')
parser.add_argument('--verbose', type=int, default=2, metavar='N',
                    help='verbose (default: 2)')
parser.add_argument('--n-train', type=int, default=1600, metavar='N',
                    help='number of training samples (default 16000')
parser.add_argument('--n-test', type=int, default=400, metavar='N',
                    help='number of test samples (default 4000')
parser.add_argument('--output-suffix', type=str, default='None', metavar='N',
                    help='suffix output filename(default: )')
parser.add_argument('--decay', type=float, default=0.004, metavar='N',
                    help='decay for the Adam optimizer (default: 0.004)')
parser.add_argument('--I-length', type=float, default=80.0, metavar='N',
                    help='Lenght of the interval (default: 80)')
args = parser.parse_args()
# parameters
decay = args.decay
intLength = args.I_length
N_epochs = args.epoch
alpha = args.alpha
L = args.L
N_cnn = args.n_cnn
lr = args.lr
batch_size = args.batch_size
inter_size = args.inter_size  # interpolate size
restr_size = args.restr_size  # restrict size
data_folder = 'data/'
log_folder = 'logs/'
models_folder = 'models/'  # folder to save the models (weights)

# this is generic for al the test_files
script__name = os.path.basename(__file__)
dir__name =  os.getcwd()
print("local directoty")
print(dir__name)

script__name = script__name.replace(".", "_")

# adding a time stamp
time_str = strftime("%Y-%m-%d_%H:%M:%S", gmtime())

outputfilename = log_folder + 'log_'+ script__name + \
                 '_Input_' + args.input_prefix +\
                 '_alpha_' + str(args.alpha) + \
                 '_L_' + str(args.L) + \
                 '_N_cnn_' + str(args.n_cnn) +\
                 '_n_train_' + str(args.n_train) + \
                 '_n_test_' + str(args.n_test);

if(args.output_suffix == 'None'):
    # by default add a time stamp
    outputfilename += '_' + time_str + '.txt'
else:
    # otherwise you can specify a suffix
    outputfilename += args.output_suffix + '.txt'
os = open(outputfilename, "w+")


if batch_size == 0:
  batch_size = args.n_train//100


def output(obj):
    print(obj)
    os.write(str(obj)+'\n')
def outputnewline():
    os.write('\n')
    os.flush()

filenameIpt = data_folder + 'Input_'  + args.input_prefix + '.h5'
filenameOpt = data_folder + 'Output_' + args.input_prefix + '.h5'

# import data: size of data: Nsamples * Nx
fInput = h5py.File(filenameIpt,'r')
InputArray = fInput['R'][:].T

fOutput = h5py.File(filenameOpt,'r')
RhoT = fOutput['Rho'][:]
Etotal = fOutput['Etotal'][:]
Ftotal = fOutput['Ftotal'][:]

Nsamples = InputArray.shape[0]
Nx = RhoT.shape[1]

output(args)
outputnewline()
output('Input data filename     = %s' % filenameIpt)
output('Output data filename    = %s' % filenameOpt)
output("Nx                      = %d" % Nx)
output("Nsamples                = %d" % Nsamples)
outputnewline()

assert RhoT.shape[0] == Nsamples
assert RhoT.shape[1] == Nx

### some pre-processing of the date
Etot_mean = np.mean(Etotal)
Etot_sigma = np.std(Etotal)

Etotal = (Etotal-Etot_mean)

n_input = InputArray.shape[1]
n_output = Nx
Natoms = n_input

# building the grid for the x data to be fed
Xpos = (np.array(range(Nx))*intLength/float(Nx))

Input2 = np.array( [[ [ np.concatenate(([r],InputArray[i,:])) ] for r in Xpos] for i in range(Nsamples) ] )
Input2 = np.reshape(Input2, (Nsamples*Nx, n_input+1))

# Output has Energy, Forces and Rho
Output2 = np.array( [[np.concatenate((Etotal[i], Ftotal[i,:], [r]))  for r in RhoT[i,:]] for i in range(Nsamples) ] )

# preraring the data both input and output
InputArrayMod = Input2
OutputArrayMod = Output2.reshape(Nsamples*Nx,Natoms+2)

# modifying the definition of the number of samples
Nsamples = Nsamples*Nx
n_train = args.n_train*Nx
n_test = args.n_test*Nx
n_input = n_input + 1
n_output = 1

# # the number of train and test samples should be less that total number
# # of samples
assert n_train+n_test <= Nsamples

# randomly reordering the augmented data
InputArrayMod  = InputArrayMod[0:n_train+n_test, :]
OutputArrayMod = OutputArrayMod[0:n_train+n_test, :]

IdxPer = np.random.permutation(n_train+n_test)
#IdxPer = np.array(range(Nsamples))

InputArray  = InputArrayMod[IdxPer, :]
OutputArray = OutputArrayMod[IdxPer, :]

xtrain = InputArray[0:n_train, :] #equal to 0:(n_train-1) in matlab
ytrain = OutputArray[0:n_train, :]

# we use the queue of
xtest  = InputArray[n_train:n_test+n_train, :] #equal to n_train:(n_train-1) or n_train:end
ytest  = OutputArray[n_train:n_test+n_train, :]

output("[n_input, n_output] = [%d, %d]" % (n_input, n_output))
output("[n_train, n_test] = [%d, %d]" % (n_train, n_test))

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
  R_IJ_H.append(tf.concat(R_Array[(Natoms-1)*i:(i+1)*(Natoms-1)], axis = 1, name= 'concat_R_IJ_H'+str(i)))


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


Nparameters = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
output("number of parameters = %d" % Nparameters)




# Using Adam optimizer with an exponential schedule
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = lr
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           Nx*n_train//batch_size, 0.95, staircase=True)

# Passing global_step to minimize() will increment it at each step.

optimizer = tf.train.AdamOptimizer(learning_rate,0.9,0.9999)
train = optimizer.minimize(cost, global_step = global_step)

# initialize the graph and the variables within
init = tf.global_variables_initializer()

saver = tf.train.Saver()

modelsavefolder = models_folder + 'model_'+ script__name + \
                 '_Input_' + args.input_prefix +\
                 '_alpha_' + str(args.alpha) + \
                 '_n_train_' + str(n_train) + \
                 '_n_test_' + str(n_test) 


train_loss = []
test_loss = []

min_test_err = 100

with tf.Session() as sess:

  sess.run(tf.global_variables_initializer())

  try:
    saver.restore(sess, dir__name+"/"+modelsavefolder+"/model.ckpt")
    print("Model restored")
    loss_test = sess.run(cost, feed_dict= {X: xtest[:,0:1],R:xtest[:,1:], Y: ytest})
    output('Test Loss from saved model: %.8e'  % loss_test)
    # setting min test loss
    min_test_err = loss_test
  except:
    pass

  for epoch in range(N_epochs):

    IdxPer = np.random.permutation(n_train)
    xtrain = xtrain[IdxPer, :]
    ytrain = ytrain[IdxPer,:]

    n_batches = n_train // batch_size
    for iteration in range(n_batches):

      # preparing the data
      x_batch = xtrain[iteration*batch_size:(iteration+1)*batch_size,0:1]
      r_batch = xtrain[iteration*batch_size:(iteration+1)*batch_size,1:]
      y_batch = ytrain[iteration*batch_size:(iteration+1)*batch_size,:]

      # running the training on the batch
      sess.run(train, feed_dict= {X: x_batch,R: r_batch, Y: y_batch})

      if iteration%(n_batches//5)==0:
        # print('Batch:', iteration, '/', n_batches)
        loss_train = sess.run(cost, feed_dict= {X: x_batch,R: r_batch, Y: y_batch})
        output('Loss: %.1e'  % loss_train)
    loss_test = sess.run(cost, feed_dict= {X: xtest[:,0:1],R:xtest[:,1:], Y: ytest})
    output('Epoch : %d and test Loss: %.1e'  % (epoch, loss_test))

    Rho_out = sess.run(Rho, feed_dict= { X: xtest[:,0:1], R: xtest[:,1:], Y: ytest})
    E_out = sess.run(Energy, feed_dict= { X: xtest[:,0:1], R: xtest[:,1:], Y: ytest})
    F_out = sess.run(Forces, feed_dict= { X: xtest[:,0:1], R: xtest[:,1:], Y: ytest})

    Rho_test = ytest[:,Natoms+1:]
    E_test = ytest[:,0:1]
    F_test = ytest[:,1:Natoms+1]

    Rho_out_norm = np.sqrt(np.sum(Rho_out**2))
    Rho_test_norm = np.sqrt(np.sum(Rho_test**2))
    Rho_test_err = np.sqrt(np.sum((Rho_test-Rho_out)**2))/Rho_test_norm

    E_out_norm = np.sqrt(np.sum(E_out**2))
    E_test_norm = np.sqrt(np.sum(E_test**2))
    E_test_err = np.sqrt(np.sum((E_test-E_out)**2))/E_test_norm

    F_out_norm = np.sqrt(np.sum(F_out**2))
    F_test_norm = np.sqrt(np.sum(F_test**2))
    F_test_err = np.sqrt(np.sum((F_test-F_out)**2))/F_test_norm

  
    output('relative error of rho %.3e'% Rho_test_err) 
    output('relative error of E %.3e'% E_test_err) 
    output('relative error of F %.3e'% F_test_err) 





    if loss_test < min_test_err:
      min_test_err = loss_test
      output("New lowest test cost reached, saving the model ")
      save_path = saver.save(sess, dir__name+"/"+modelsavefolder+"/model.ckpt")
      output("Model saved in path: %s" % save_path)


# possible strategies
# normalize the energy (it is very big compared to the rest)
# learn the energy first along with rho, and at the end use taht to lear the forces