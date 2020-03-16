import os, json
from copy import deepcopy

def save_train_config(config_name, config_dict, config_base):
  # json.dumps(config_dict)
  config_folder = 'experiment_config/{}'.format(config_base)
  os.makedirs(config_folder, exist_ok=True)
  with open('{}/{}.json'.format(config_folder, config_name), 'w') as jsfile:
    json.dump(config_dict, jsfile, indent=2) # indent forces pretty printing

## train_config = {'model_kwargs':, 'train_kwargs', 'data_kwargs':}

def param_str(v):
  if type(v) in [list, tuple]:
    return '-'.join([str(o) for o in v])
  else:
    return str(v)

def save_experiment_params(base_config_dict, params, expt_set):
  """
  expt_set is basically just the name of the directory in experiment_config within which 
           json will be saved. Sharing a common expt_set is a way of grouping sets of related experiments.
  """
  shorthand_sets = {'tk': 'train_kwargs', 'dk': 'data_kwargs',
                    'mk': 'model_kwargs', 'vk': 'val_kwargs'}
  for plist in params:
    config = deepcopy(base_config_dict)
    pv = []
    for s, param, v in plist:
      config[shorthand_sets.get(s, s)][param] = v
      print(param)
      if type(v) == dict:
        pv.append(param+'-'.join([str(k)+param_str(item) for k, item in v.items()]))
      else:
        pv.append(param+param_str(v))

    config['expt_name'] = expt_set+'_'+'-'.join(pv)
    save_train_config('-'.join(pv), config, expt_set)

def save_train_config(expt_set, expt_name, model_class, data_loader,
                      weighted_average=False, eval_freq=1000000,
                      train_kwargs={}):
  base_kwargs = {}
  base_kwargs['model_kwargs'] = model_class.config
  base_kwargs['model_kwargs']['model_class'] = model_class.__class__.__name__
  base_kwargs['data_kwargs'] = data_loader.config
  base_kwargs['data_kwargs']['data_class'] = data_loader.__class__.__name__
  base_kwargs['val_kwargs'] = {'weighted_average': weighted_average, 'eval_freq': eval_freq}
  base_kwargs['train_kwargs'] = train_kwargs

  config_output_path = config_dir + expt_set
  os.makedirs(config_output_path, exist_ok=True)
  with open(config_output_path + '/' + expt_name +'.json', 'w') as outf:
    json.dump(base_kwargs, outf, indent=2)