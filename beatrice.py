from absl import app
from absl import flags
import os
import scripts.trainer as trainer
FLAGS = flags.FLAGS

#  Define Flagg names for inputs.
flags.DEFINE_string('z', 'example_data/Simulation_data0.z', 'Location of Z Score')
flags.DEFINE_string('target', 'results', 'Location to store results.')
flags.DEFINE_string('LD', 'example_data/Simulation_data0.ld', 'Location of LD matrix')
flags.DEFINE_string('prior_location', '', 'Location where priors of the underlying probability map of bianry concrete distribution is stored.')
flags.DEFINE_integer('N', 5000, 'Number of subjects.', lower_bound=0)
flags.DEFINE_integer('MCMC_samples', 1, 'Number of random samples for MC integration', lower_bound=1)
flags.DEFINE_integer('max_iter', 2001, 'Number of training iterations.',lower_bound=500)
flags.DEFINE_boolean('plot_loss', True, 'Plot training losses.')
flags.DEFINE_boolean('get_cred', True, 'Get Credible Sets')
flags.DEFINE_float('gamma', 0.1, 'Threshold to create the reduced space of binary vectors B^R.')
flags.DEFINE_float('gamma_key', 0.2, 'Threshold for key variants.')
flags.DEFINE_float('gamma_coverage', 0.95, 'Threshold for coverage.')
flags.DEFINE_float('gamma_selection', 0.05, 'Threshold for selection probability within a credible set.')
flags.DEFINE_float('sigma_sq', 0.05, 'Variance of causal variants')
flags.DEFINE_float('temp_lower_bound', 0.01, 'Extent of continuous relaxations', lower_bound=0.005)
flags.DEFINE_integer('sparse_concrete', 50, 'Number of non zero locatons of the concrete random vector at every iteration.', lower_bound=49)
flags.DEFINE_list('true_loc', '', 'Index of true causal variants.')


def main(argv):
    if not os.path.exists(FLAGS.z):
        print('Location of Z doesn\'t exist: ' , FLAGS.z)
        return
  
    if not os.path.exists(FLAGS.LD):
        print('Location of LD doesn\'t exist: ' , FLAGS.LD)
        return
    
    if not os.path.exists(FLAGS.target):
        os.makedirs(FLAGS.target)
    options = {}    
    options['target'] = FLAGS.target
    options['z'] = FLAGS.z
    options['LD'] = FLAGS.LD
    options['n_sub'] = FLAGS.N
    options['loc_true'] = [int(i) for i in FLAGS.true_loc]
    options['MCMC_samples'] = FLAGS.MCMC_samples
    options['sigma_sq'] = FLAGS.sigma_sq
    options['max_iter'] = FLAGS.max_iter
    options['temp_lower_bound'] = FLAGS.temp_lower_bound
    options['prior_location'] = FLAGS.prior_location
    options['plot_loss'] = FLAGS.plot_loss
    options['coverage_ths'] = FLAGS.gamma_coverage
    options['selection_prob'] = FLAGS.gamma_selection
    options['key_thres'] = FLAGS.gamma_key
    options['get_cred'] = FLAGS.get_cred
    options['sparsity_cl'] = FLAGS.sparse_concrete
    options['gamma'] = FLAGS.gamma
    trainer.main(options)
    
if __name__ == '__main__':
  app.run(main)
