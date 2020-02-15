import tensorflow as tf
import gym

import atari_zoo
from atari_zoo import game_list
from atari_zoo import MakeAtariModel
import atari_zoo.atari_wrappers as atari_wrappers
from atari_zoo.dopamine_preprocessing import AtariPreprocessing as DopamineAtariPreprocessing	
from atari_zoo.atari_wrappers import FireResetEnv, NoopResetEnv, MaxAndSkipEnv,WarpFrameTF,FrameStack,ScaledFloatFrame
from atari_zoo.model_maker import RL_model
from atari_zoo.config import datadir_local_dict,datadir_remote_dict,url_formatter_dict 
from atari_zoo import game_action_counts 
import atari_zoo.log



#Ape-X (recent high-performing DQN variant)
class RL_Apex(RL_model):
  channel_order = "NHWC" #"NCHW"

  #note: action_value/Relu also worth considering...
  layers = [
     {'type': 'conv', 'name': 'deepq/q_func/convnet/Conv/Relu', 'size': 32},
     {'type': 'conv', 'name': 'deepq/q_func/convnet/Conv_1/Relu', 'size': 64},
     {'type': 'conv', 'name': 'deepq/q_func/convnet/Conv_2/Relu', 'size': 64},
     {'type': 'dense', 'name': 'deepq/q_func/state_value/Relu', 'size': 512},
     {'type': 'dense', 'name': 'deepq/q_func/q_values', 'size':18}
   ]

  weights = [
      {'name':'deepq/q_func/convnet/Conv/weights'},
      {'name':'deepq/q_func/convnet/Conv_1/weights'},
      {'name':'deepq/q_func/convnet/Conv_2/weights'}
  ]

  def get_action(self,model):
        policy = model(self.layers[-1]['name']) #"a2c/policy/BiasAdd")
        action_sample = tf.argmax(policy, axis=-1)
        return action_sample, policy


#Uber's Deep GA
class RL_GA(RL_model):
  layers = [
     {'type': 'conv', 'name': 'ga/conv1/relu', 'size': 32},
     {'type': 'conv', 'name': 'ga/conv2/relu', 'size': 64},
     {'type': 'conv', 'name': 'ga/conv3/relu', 'size': 64},
     {'type': 'dense', 'name': 'ga/fc/relu', 'size': 512},
     {'type': 'dense', 'name': 'ga/out/signal', 'size':18}
   ]

  weights = [
      {'name':'ga/conv1/w'},
      {'name':'ga/conv2/w'},
      {'name':'ga/conv3/w'},
  ]

  def get_action(self,model):
        policy = model(self.layers[-1]['name'])
        action_sample = tf.argmax(policy, axis=-1)
        return action_sample, policy

  def preprocess_weight(self,x):
      return x[0]


""" Rainbow & DQN"""

#Rainbow (slightly older high-performing DQN variant)
class RL_Rainbow_dopamine(RL_model):
  #ph_type = 'uint8'
  valid_run_range = (1,5)
  preprocess_style = 'dopamine'
  input_scale = 255.0
  image_value_range = (0, 255) 
  #input_name = 'state_ph'
  input_name = 'Online/Cast'

  weights = [
      {'name':'Online/Conv/weights'},
      {'name':'Online/Conv_1/weights'},
      {'name':'Online/Conv_2/weights'}
  ]

  layers = [
     {'type': 'conv', 'name': 'Online/Conv/Relu', 'size': 32},
     {'type': 'conv', 'name': 'Online/Conv_1/Relu', 'size': 64},
     {'type': 'conv', 'name': 'Online/Conv_2/Relu', 'size': 64},
     {'type': 'dense', 'name': 'Online/fully_connected/Relu', 'size': 512}, 
     {'type': 'dense', 'name': 'Online/fully_connected_1/BiasAdd', 'size':18},
     {'type': 'dense', 'name': 'Online/Sum', 'size':18}
   ]

  additional_layers={'c51':{'type':'dense','name': 'Online/fully_connected_1/BiasAdd', 'size':18*51}}

 
  def get_action(self,model):
      policy = model(self.layers[-1]['name'])      
      action_sample = tf.argmax(policy, axis=1)
      return action_sample, policy

  def get_log(self):
    raise NotImplementedError
    #"Integration with Dopamine log formatting not yet complete."

  def get_checkpoint_info(self):
    raise NotImplementedError
    #"Dopamine models include only the final checkpoint."


#DQN from dopamine model dump
class RL_DQN_dopamine(RL_model):
  #ph_type = 'uint8'
  input_scale = 255.0
  preprocess_style = 'dopamine'
  image_value_range = (0, 255) 
  input_name = 'Online/Cast'
  valid_run_range = (1,3)

  weights = [
      {'name':'Online/Conv/weights'},
      {'name':'Online/Conv_1/weights'},
      {'name':'Online/Conv_2/weights'}
  ]

  layers = [
     {'type': 'conv', 'name': 'Online/Conv/Relu', 'size': 32},
     {'type': 'conv', 'name': 'Online/Conv_1/Relu', 'size': 64},
     {'type': 'conv', 'name': 'Online/Conv_2/Relu', 'size': 64},
     {'type': 'dense', 'name': 'Online/fully_connected/Relu', 'size': 512},
     {'type': 'dense', 'name': 'Online/fully_connected_1/BiasAdd', 'size':18}
   ]
 
  def get_action(self,model):
      policy = model(self.layers[-1]['name'])      
      action_sample = tf.argmax(policy, axis=1)
      return action_sample, policy

      # policy = model(self.layers[-1]['name']) 
      # action_sample = tf.argmax(policy, axis=1)
      # return action_sample

  def get_log(self):
    raise NotImplementedError
      #Integration with Dopamine log formatting not yet complete."

  def get_checkpoint_info(self):
    raise NotImplementedError
       #,"Dopamine models include only the final checkpoint."


class RL_A2C(RL_model):
  weights = [
      {'name':'a2c/conv1/weights'},
      {'name':'a2c/conv2/weights'},
      {'name':'a2c/conv3/weights'}
  ]

  layers = [
     {'type': 'conv', 'name': 'a2c/conv1/Relu', 'size': 32},
     {'type': 'conv', 'name': 'a2c/conv2/Relu', 'size': 64},
     {'type': 'conv', 'name': 'a2c/conv3/Relu', 'size': 64},
     {'type': 'dense', 'name': 'a2c/fc/Relu', 'size': 512},
     #TODO: enable accesing a2c's value head as well! 
     #{'type': 'dense', 'name': 'a2c/value/BiasAdd', 'size':18},
     {'type': 'dense', 'name': 'a2c/policy/BiasAdd', 'size':18}
   ]
  
  def get_action(self,model):
        policy = model(self.layers[-1]['name']) 
        rand_u = tf.random_uniform(tf.shape(policy))
        action_sample = tf.argmax(policy - tf.log(-tf.log(rand_u)), axis=-1)
        return action_sample, policy

class RL_IMPALA(RL_model):
  input_name = 'agent_1/agent/unroll/batch_apply/truediv'
  preprocess_style = 'np'

  weights = [
      {'name':'agent/batch_apply/convnet/conv_2d/w'},
      {'name':'agent/batch_apply/convnet/conv_2d_1/w'},
      {'name':'agent/batch_apply/convnet/conv_2d_2/w'},
  ]

  layers = [
     {'type': 'conv', 'name': 'agent_1/agent/unroll/batch_apply/convnet/Relu', 'size': 32},
     {'type': 'conv', 'name': 'agent_1/agent/unroll/batch_apply/convnet/Relu_1', 'size': 64},
     {'type': 'conv', 'name': 'agent_1/agent/unroll/batch_apply/convnet/Relu_2', 'size': 64},
     {'type': 'dense', 'name': 'agent_1/agent/unroll/batch_apply/Relu', 'size': 512},
     {'type': 'dense', 'name': 'agent_1/agent/unroll/batch_apply_1/policy_logits/add', 'size': 18},
   ]

  def get_action(self,model):
        policy_logits = model(self.layers[-1]['name'])
        new_action = tf.multinomial(policy_logits, num_samples=1,
                output_dtype=tf.int32)
        new_action = tf.squeeze(new_action, 1, name='new_action')
        return new_action, policy_logits


class RL_ES(RL_model):
  weights = [
      {'name':'es/layer1/conv1/w'},
      {'name':'es/layer2/conv2/w'},
      {'name':'es/layer3/conv3/w'},
  ]

  layers = [
     {'type': 'conv', 'name': 'es/layer1/Relu', 'size': 32},
     {'type': 'conv', 'name': 'es/layer2/Relu', 'size': 64},
     {'type': 'conv', 'name': 'es/layer3/Relu', 'size': 64},
     {'type': 'dense', 'name': 'es/layer4/Relu', 'size': 512},
     {'type': 'dense', 'name': 'es/layer5/out/out', 'size':18}
   ]

  def preprocess_weight(self,x):
      return x[0]

  def get_action(self,model):
        policy = model(self.layers[-1]['name']) 
        action_sample = tf.argmax(policy, axis=-1)
        return action_sample, policy


""" Other models """

### Instantiate concrete models using python magic
#class_map = {'ga':RL_GA,'es':RL_ES,'apex':RL_Apex,'a2c':RL_A2C,'dqn':RL_DQN_dopamine,'rainbow':RL_Rainbow_dopamine, 'impala':RL_IMPALA}
class_map = {'rainbow':RL_Rainbow_dopamine, 'dqn': RL_DQN_dopamine, 'ga': RL_GA, 'a2c': RL_A2C, 'impala': RL_IMPALA, 'es': RL_ES, 'apex': RL_Apex}

#helper utility to make new python model classes
def _MakeAtariModel(model_class,name,environment,model_path,run_id,algorithm,log_path,data_path):
    #find number of actions in this particular game
    num_actions = game_action_counts[environment]

    #change last layer size to reflect available actions
    #layers = model_class.layers.copy()
    layers = list(model_class.layers) #python2.7 compatibility
    layers[-1]['size']=num_actions
    #create inherited class with correct properties (hack?)
    return type('Atari'+name,(model_class,),{'model_path':model_path,'environment':environment,'layers':layers,'run_id':run_id,'algorithm':algorithm,
                                             'log_path':log_path,'data_path':data_path})

"""
Helper function to get paths to model, rollout data, and log
for a particular algo/env/run combo
"""
def GetFilePathsForModel(algo,environment,run_no,tag='final',local=False):

    #if loading off of local disk (rare; only for development)
    if local:
        data_root = datadir_local_dict[algo]
        if tag==None:
            model_path = "%s/%s/model%d.pb" % (data_root,environment,run_no)
            data_path = "%s/%s/model%d_rollout.npz" % (data_root,environment,run_no)
        else:
            model_path = "%s/%s/model%d_%s.pb" % (data_root,environment,run_no,tag)
            data_path = "%s/%s/model%d_%s_rollout.npz" % (data_root,environment,run_no,tag)

        log_path = "%s/checkpoints/%s_%d" % (data_root,environment,run_no)

    #otherwise if loading off the canonical remote server (most common)
    else:
        data_root = datadir_remote_dict[algo]
        if tag==None:
            model_path = "%s/%s/model%d.pb" % (data_root,environment,run_no)
            data_path = "%s/%s/model%d_rollout.npz" % (data_root,environment,run_no)
        else:
            model_path = "%s/%s/model%d_%s.pb" % (data_root,environment,run_no,tag)
            data_path = "%s/%s/model%d_%s_rollout.npz" % (data_root,environment,run_no,tag)

        if (algo,'remote') in url_formatter_dict:
            model_path = url_formatter_dict[(algo,'remote')](data_root,algo,environment,run_no)
   
        log_path = "%s/checkpoints/%s_%d" % (data_root,environment,run_no)

    return model_path,data_path,log_path

"""
Function to query for available checkpoints for a model
"""
def GetAvailableTaggedCheckpoints(algo,environment,run_no,local=False):
    _,_,log_path = GetFilePathsForModel(algo,environment,run_no,local=local)
    json_data = atari_zoo.log.load_checkpoint_info(log_path) 
    chkpoint_info = atari_zoo.log.parse_checkpoint_info(json_data)
    return chkpoint_info



"""
Function to load model from the model zoo
algo: Algorithm (ga,es,apex,a2c,dqn,rainbow)
environment: Atari gym environment (e.g. SeaquestNoFrameskip-v4)
run_no: which run of the algorithm
tag: which tag to search
local: boolean, whether to get the model from a local archive or from the remote server
"""
def MakeAtariModel(algo,environment,run_no,tag='final',local=False):

    model_path,data_path,log_path = GetFilePathsForModel(algo,environment,run_no,tag,local)

    # if atari_zoo.config.debug:
    #     print('Model path:',model_path)
    #     print('Data path:',data_path)
    #     print('Log path:',log_path)

    name = "%s_%s_%d_%s" % (algo,environment,run_no,tag)

    model_class = class_map[algo]
   
    valid_run_range = model_class.valid_run_range
    if run_no < valid_run_range[0] or run_no > valid_run_range[1]:
        raise ValueError("Requested run %d out of range (%d,%d)"%(run_no,valid_run_range[0],valid_run_range[1]))

    return _MakeAtariModel(class_map[algo],name,environment,model_path,run_no,algo,log_path,data_path)
