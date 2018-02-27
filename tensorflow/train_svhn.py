import logging
from datetime import datetime

from experiments.run_context import RunContext
from datasets import SVHN
from mean_teacher.model import Model
from mean_teacher import minibatching


logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger('main')


def run():
    data_seed = 0
    n_labeled = 500
    n_labeled_per_batch = 10
    n_extra_unlabeled = 0
    kfac_sampling_type = 'sghmc'

    model = Model(RunContext(__file__, 0), kfac_sampling_type=kfac_sampling_type)
    model['n_labeled'] = n_labeled
    model['regularization_weight'] = 0.0
    model['ema_consistency'] = 0
    #model['student_dropout_probability'] = 0.5
    #model['teacher_dropout_probability'] = 0.5
    #model['max_consistency_cost'] = 0.0
    #model['apply_consistency_to_labeled'] = False
    model['ema_decay_during_rampup'] = 0.6#0.95
    model['ema_decay_after_rampup'] = 0.6#0.95
    model['rampup_length'] = 40000
    model['rampdown_length'] = 10000
    model['training_length'] = 180000
    model['print_span'] = 1
    model['evaluation_span'] = 100
    #model['rampdown_length'] = 10000#0
    #model['training_length'] = 40000#180000
    model['kfac_inv_update_span'] = 100
    model['kfac_noise_learning_factor'] = 1.0
    model['kfac_damping'] = 1.0
    model['kfac_norm_constraint'] = 0.0001
    model['adam_beta_1_before_rampdown'] = 0.9#0.99#0.9
    model['adam_beta_1_after_rampdown'] = 0.5#0.99#0.5

    tensorboard_dir = model.save_tensorboard_graph()
    LOG.info("Saved tensorboard graph to %r", tensorboard_dir)

    svhn = SVHN(data_seed, n_labeled, n_extra_unlabeled)
    training_batches = minibatching.training_batches(svhn.training, n_labeled_per_batch=n_labeled_per_batch)
    evaluation_batches_fn = minibatching.evaluation_epoch_generator(svhn.evaluation)

    model.train(training_batches, evaluation_batches_fn)


if __name__ == "__main__":
    run()
