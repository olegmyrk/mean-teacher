"Mean teacher model"

import logging
import os
from collections import namedtuple

import numpy as np

import tensorflow as tf
from tensorflow.contrib import metrics, slim
from tensorflow.contrib.metrics import streaming_mean

import tensorflow.contrib.kfac

from . import nn
from . import weight_norm_no_batchnorm as wn
from .framework import ema_variable_scope, name_variable_scope, assert_shape, HyperparamVariables
from . import string_utils


LOG = logging.getLogger('main')


class Model:
    DEFAULT_HYPERPARAMS = {
        # Consistency hyperparameters
        'ema_consistency': 1,
        'apply_consistency_to_labeled': True,
        'max_consistency_cost': 100.0,
        'ema_decay_during_rampup': 0.99,
        'ema_decay_after_rampup': 0.999,
        'consistency_trust': 0.0,
        'num_logits': 1, # Either 1 or 2
        'logit_distance_cost': 0.0, # Matters only with 2 outputs

        # Optimizer hyperparameters
        'max_learning_rate': 0.003,
        'adam_beta_1_before_rampdown': 0.9,
        'adam_beta_1_after_rampdown': 0.5,
        'adam_beta_2_during_rampup': 0.99,
        'adam_beta_2_after_rampup': 0.999,
        'adam_epsilon': 1e-8,

        # Architecture hyperparameters
        'input_noise': 0.15,
        'student_dropout_probability': 0.5,
        'teacher_dropout_probability': 0.5,
        'regularization_weight' : 1.0,
        'n_labeled' : 0,

        # Training schedule
        'rampup_length': 40000,
        'rampdown_length': 25000,
        'training_length': 150000,

        # Input augmentation
        'flip_horizontally': False,
        'translate': True,

        # Whether to scale each input image to mean=0 and std=1 per channel
        # Use False if input is already normalized in some other way
        'normalize_input': True,

        # Output schedule
        'print_span': 20,
        'evaluation_span': 500,

        # KFAC parameters 
        'kfac_inv_update_span' : 100,
        'kfac_noise_learning_factor' : 0.0,
        'kfac_damping' : 0.001,
        'kfac_norm_constraint' : 0.0001
    }

    #pylint: disable=too-many-instance-attributes
    def __init__(self, run_context=None, kfac_sampling_type = None):
        if run_context is not None:
            self.training_log = run_context.create_train_log('training')
            self.validation_log = run_context.create_train_log('validation')
            self.checkpoint_path = os.path.join(run_context.transient_dir, 'checkpoint')
            self.tensorboard_path = os.path.join(run_context.result_dir, 'tensorboard')

        with tf.name_scope("placeholders"):
            self.images = tf.placeholder(dtype=tf.float32, shape=(None, 32, 32, 3), name='images')
            self.labels = tf.placeholder(dtype=tf.int32, shape=(None,), name='labels')
            self.ensemble_probs = tf.placeholder(dtype=tf.float32, shape=(None,10), name='ensemble_probs')
            self.is_training = tf.placeholder(dtype=tf.bool, shape=(), name='is_training')

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        tf.add_to_collection("init_in_init", self.global_step)
        self.hyper = HyperparamVariables(self.DEFAULT_HYPERPARAMS)
        for var in self.hyper.variables.values():
            tf.add_to_collection("init_in_init", var)

        with tf.name_scope("ramps"):
            sigmoid_rampup_value = sigmoid_rampup(self.global_step, self.hyper['rampup_length'])
            sigmoid_rampdown_value = sigmoid_rampdown(self.global_step,
                                                      self.hyper['rampdown_length'],
                                                      self.hyper['training_length'])
            self.learning_rate = tf.multiply(sigmoid_rampup_value * sigmoid_rampdown_value,
                                             self.hyper['max_learning_rate'],
                                             name='learning_rate')
            self.adam_beta_1 = tf.add(sigmoid_rampdown_value * self.hyper['adam_beta_1_before_rampdown'],
                                      (1 - sigmoid_rampdown_value) * self.hyper['adam_beta_1_after_rampdown'],
                                      name='adam_beta_1')
            self.cons_coefficient = tf.multiply(sigmoid_rampup_value,
                                                self.hyper['max_consistency_cost'],
                                                name='consistency_coefficient')

            step_rampup_value = step_rampup(self.global_step, self.hyper['rampup_length'])
            self.adam_beta_2 = tf.add((1 - step_rampup_value) * self.hyper['adam_beta_2_during_rampup'],
                                      step_rampup_value * self.hyper['adam_beta_2_after_rampup'],
                                      name='adam_beta_2')
            self.ema_decay = tf.add((1 - step_rampup_value) * self.hyper['ema_decay_during_rampup'],
                                    step_rampup_value * self.hyper['ema_decay_after_rampup'],
                                    name='ema_decay')

        layer_collection = tensorflow.contrib.kfac.layer_collection.LayerCollection()
        (
            self.train_init_pass,
            (self.class_logits_1, self.cons_logits_1),
            (self.class_logits_2, self.cons_logits_2),
            (self.class_logits_ema, self.cons_logits_ema)
        ) = inference(
            self.images,
            layer_collection=layer_collection,
            is_training=self.is_training,
            ema_decay=self.ema_decay,
            input_noise=self.hyper['input_noise'],
            student_dropout_probability=self.hyper['student_dropout_probability'],
            teacher_dropout_probability=self.hyper['teacher_dropout_probability'],
            normalize_input=self.hyper['normalize_input'],
            flip_horizontally=self.hyper['flip_horizontally'],
            translate=self.hyper['translate'],
            num_logits=self.hyper['num_logits'])

        layer_collection.register_categorical_predictive_distribution(self.class_logits_1, name="class_logits_1")

        with tf.name_scope("objectives"):
            self.mean_error_1, self.errors_1 = errors(self.class_logits_1, self.labels)
            self.mean_error_ema, self.errors_ema = errors(self.class_logits_ema, self.labels)

            self.mean_class_cost_1, self.class_costs_1 = classification_costs(
                self.class_logits_1, self.labels)
            self.class_probs_1 = tf.nn.softmax(self.class_logits_1)
            self.mean_class_cost_ema, self.class_costs_ema = classification_costs(
                self.class_logits_ema, self.labels)

            labeled_consistency = self.hyper['apply_consistency_to_labeled']
            consistency_mask = tf.logical_or(tf.equal(self.labels, -1), labeled_consistency)
            self.mean_cons_cost_pi, self.cons_costs_pi = consistency_costs(
                self.cons_logits_1, self.class_logits_2, self.cons_coefficient, consistency_mask, self.hyper['consistency_trust'])
            self.mean_cons_cost_mt, self.cons_costs_mt = consistency_costs(
                self.cons_logits_1, self.class_logits_ema, self.cons_coefficient, consistency_mask, self.hyper['consistency_trust'])
            self.mean_cons_cost_ens, self.cons_costs_ens = ensemble_classification_costs(self.class_logits_1, self.ensemble_probs, self.cons_coefficient, consistency_mask)

            def l2_norms(matrix):
                l2s = tf.reduce_sum(matrix ** 2, axis=1)
                mean_l2 = tf.reduce_mean(l2s)
                return mean_l2, l2s

            self.mean_res_l2_1, self.res_l2s_1 = l2_norms(self.class_logits_1 - self.cons_logits_1)
            self.mean_res_l2_ema, self.res_l2s_ema = l2_norms(self.class_logits_ema - self.cons_logits_ema)
            self.res_costs_1 = self.hyper['logit_distance_cost'] * self.res_l2s_1
            self.mean_res_cost_1 = tf.reduce_mean(self.res_costs_1)
            self.res_costs_ema = self.hyper['logit_distance_cost'] * self.res_l2s_ema
            self.mean_res_cost_ema = tf.reduce_mean(self.res_costs_ema)

            self.mean_total_cost_pi, self.total_costs_pi = total_costs(
                self.class_costs_1, self.cons_costs_pi, self.res_costs_1)
            self.mean_total_cost_mt, self.total_costs_mt = total_costs(
                self.class_costs_1, self.cons_costs_mt, self.res_costs_1)
            self.mean_total_cost_ens, self.total_costs_ens = total_costs(
                self.class_costs_1, self.cons_costs_ens, self.res_costs_1)
            self.mean_total_cost_vanilla, self.total_costs_vanilla = total_costs(
                self.class_costs_1, self.res_costs_1)

            self.cost_to_be_minimized = tf.cond(tf.equal(self.hyper['ema_consistency'],0),
                                                lambda: self.mean_total_cost_pi,
                                                lambda: tf.cond(tf.equal(self.hyper['ema_consistency'],1),
                                                    lambda: self.mean_total_cost_mt,
                                                    lambda: tf.cond(tf.equal(self.hyper['ema_consistency'],2),
                                                        lambda: self.mean_total_cost_ens,
                                                        lambda: self.mean_total_cost_vanilla 
                                                        )
                                                    )
                                                )

            self.labels_in_batch = tf.reduce_sum(tf.cast(tf.not_equal(self.labels, -1),dtype=tf.int32), name="labels_in_batch")
            self.batch_ratio = tf.cast(self.labels_in_batch,dtype=tf.float32) / tf.cast(self.hyper['n_labeled'],dtype=tf.float32)
            self.regularization_cost = self.batch_ratio * self.hyper['regularization_weight'] * tf.losses.get_regularization_loss()
            self.cost_to_be_minimized = self.cost_to_be_minimized + self.regularization_cost 

        with tf.name_scope("train_step"):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                #self.train_step_op = nn.adam_optimizer(self.cost_to_be_minimized,
                #                                       self.global_step,
                #                                       learning_rate=self.learning_rate,
                #                                       beta1=self.adam_beta_1,
                #                                       beta2=self.adam_beta_2,
                #                                       epsilon=self.hyper['adam_epsilon'])
                
                # Kfac must always compute eigen-decomposition 
                tensorflow.contrib.kfac.fisher_factors.set_global_constants(eigenvalue_decomposition_threshold=0)

                self.labels_in_batch = tf.reduce_sum(tf.cast(tf.not_equal(self.labels, -1),dtype=tf.int32), name="labels_in_batch")
                self.momentum = self.adam_beta_1
                self.cov_ema_decay = self.adam_beta_2
                
                self.batch_ratio = tf.cast(self.labels_in_batch,dtype=tf.float32) / tf.cast(self.hyper['kfac_n_labeled'],dtype=tf.float32)
                self.noise_learning_scale = self.hyper['kfac_noise_learning_factor'] * self.batch_ratio 

                from . import natural_gradient_optimizer
                optimizer = natural_gradient_optimizer.NaturalGradientOptimizer(
                                      var_list = layer_collection.registered_variables,
                                      main_learning_rate=self.learning_rate, #0.0001,
                                      noise_learning_scale=self.noise_learning_scale, #0.0001*50/512,
                                      cov_ema_decay=self.cov_ema_decay,#0.99,#0.95,
                                      damping=self.hyper['kfac_damping'],#0.001,#1.0,
                                      norm_constraint=self.hyper['kfac_norm_constraint'],#0.0001,#None,#0.00001,
                                      layer_collection=layer_collection,
                                      sampling_type=kfac_sampling_type,#'sghmc',
                                      momentum=self.momentum #0.9
                                      )

                #self.regularization = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1.0), [W for [W,b] in layer_collection.fisher_blocks.keys()])
                self.regularization = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1.0), [V for V in layer_collection.fisher_blocks.keys()])
                self.cost_to_be_minimized = self.cost_to_be_minimized + self.hyper['regularization_weight'] * self.regularization * self.batch_ratio
 
                # Preparing operations
                self.cov_update_op = tf.group(tf.Print(self.global_step,[self.global_step],"Update KFAC COV"), optimizer.cov_update_op)
                self.inv_update_op = tf.group(tf.Print(self.global_step,[self.global_step],"Inverting KFAC FM"), optimizer.inv_update_op)
                with tf.control_dependencies([tf.Print(self.global_step, [self.learning_rate, self.noise_learning_scale], "Learning rates: "), tf.Print(self.global_step, [self.momentum], "Momentum: "), tf.Print(self.global_step, [self.cov_ema_decay], "Cov ema decay: ")]):
                    self.train_step_op = optimizer.minimize(
                                           self.cost_to_be_minimized, 
                                           global_step=self.global_step
                                       )

        self.training_control = training_control(self.global_step,
                                                 self.hyper['print_span'],
                                                 self.hyper['evaluation_span'],
                                                 self.hyper['training_length'])

        self.training_metrics = {
            "learning_rate": self.learning_rate,
            "adam_beta_1": self.adam_beta_1,
            "adam_beta_2": self.adam_beta_2,
            "ema_decay": self.ema_decay,
            "cons_coefficient": self.cons_coefficient,
            "train/error/1": self.mean_error_1,
            "train/error/ema": self.mean_error_ema,
            "train/class_cost/1": self.mean_class_cost_1,
            "train/class_cost/ema": self.mean_class_cost_ema,
            "train/cons_cost/pi": self.mean_cons_cost_pi,
            "train/cons_cost/mt": self.mean_cons_cost_mt,
            "train/cons_cost/ens": self.mean_cons_cost_ens,
            "train/reg_cost" : self.regularization_cost,
            "train/res_cost/1": self.mean_res_cost_1,
            "train/res_cost/ema": self.mean_res_cost_ema,
            "train/total_cost/pi": self.mean_total_cost_pi,
            "train/total_cost/mt": self.mean_total_cost_mt,
        }

        with tf.variable_scope("validation_metrics") as metrics_scope:
            self.metric_values, self.metric_update_ops = metrics.aggregate_metric_map({
                "eval/error/1": streaming_mean(self.errors_1),
                "eval/error/ema": streaming_mean(self.errors_ema),
                "eval/class_cost/1": streaming_mean(self.class_costs_1),
                "eval/class_cost/ema": streaming_mean(self.class_costs_ema),
                "eval/res_cost/1": streaming_mean(self.res_costs_1),
                "eval/res_cost/ema": streaming_mean(self.res_costs_ema),
            })
            metric_variables = slim.get_local_variables(scope=metrics_scope.name)
            self.metric_init_op = tf.variables_initializer(metric_variables)

        self.result_formatter = string_utils.DictFormatter(
            order=["eval/error/ema", "eval/class_cost/ema", "error/1", "class_cost/1", "cons_cost/mt", "cons_cost/pi", "cons_cost/ens", "reg_cost"],
            default_format='{name}: {value:>10.6f}',
            separator=",  ")
        self.result_formatter.add_format('error', '{name}: {value:>6.1%}')

        with tf.name_scope("initializers"):
            init_init_variables = tf.get_collection("init_in_init")
            train_init_variables = [
                var for var in tf.global_variables() if var not in init_init_variables
            ]
            self.init_init_op = tf.variables_initializer(init_init_variables)
            self.train_init_op = tf.variables_initializer(train_init_variables)

        self.saver = tf.train.Saver()
        #self.session = tf.Session()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)
        self.run(self.init_init_op)

    def __setitem__(self, key, value):
        self.hyper.assign(self.session, key, value)

    def __getitem__(self, key):
        return self.hyper.get(self.session, key)

    def train(self, training_batches, evaluation_batches_fn):
        feed_dict = self.feed_dict(next(training_batches))
        self.run(self.train_init_op)
        self.run(self.train_init_pass, feed_dict)
        LOG.info("Model variables initialized")
        self.evaluate(evaluation_batches_fn)
        self.save_checkpoint()

        # begin temporal ensembling training #
        ensemble_ema_probs = {}
        ensemble_ema_counts = {}
        ensemble_ema_decay = self.run(self.ema_decay)
        # end temporal ensembling training #

        for batch in training_batches:
<<<<<<< Updated upstream
            # begin temporal ensembling training #
            batch_sample_ids = batch['sample_id']
            batch_labels = batch['y']
            batch_ensemble_probs = np.zeros((len(batch_sample_ids), 10), np.float32)

            for (batch_sample_id, batch_sample_idx) in zip(batch_sample_ids, range(len(batch_sample_ids))):
                if batch_sample_id in ensemble_ema_counts:
                    batch_ensemble_probs[batch_sample_idx, :] = ensemble_ema_probs[batch_sample_id] / (1 - ensemble_ema_decay ** ensemble_ema_counts[batch_sample_id])
            # end temporal ensembling training #

            batch_result_probs, results, _, _  = self.run([self.class_probs_1, self.training_metrics, self.train_step_op, self.cov_update_op],
                    self.feed_dict(batch, extra = { self.ensemble_probs : batch_ensemble_probs }))

            # begin temporal ensembling training #
            for (batch_sample_id, batch_sample_idx) in zip(batch_sample_ids, range(len(batch_sample_ids))):
                if batch_labels[batch_sample_idx] != -1:
                    continue
                if batch_sample_id in ensemble_ema_counts:
                    ensemble_ema_probs[batch_sample_id] = ensemble_ema_decay * ensemble_ema_probs[batch_sample_id] + (1 - ensemble_ema_decay) * batch_result_probs[batch_sample_idx, :]
                    ensemble_ema_counts[batch_sample_id] += 1
                else:
                    ensemble_ema_probs[batch_sample_id] = (1 - ensemble_ema_decay) * batch_result_probs[batch_sample_idx, :]
                    ensemble_ema_counts[batch_sample_id] = 1
            # end temporal ensembling training #

            step_control = self.get_training_control()
            if step_control['step'] % self.run([self.hyper['kfac_inv_update_span']])[0] == 0:
                self.run([self.inv_update_op])
            self.training_log.record(step_control['step'], {**results, **step_control})
            if step_control['time_to_print']:
                LOG.info("step %5d:   %s", step_control['step'], self.result_formatter.format_dict(results))
            if step_control['time_to_stop']:
                break
            if step_control['time_to_evaluate']:
                self.evaluate(evaluation_batches_fn)
                self.save_checkpoint()
        self.evaluate(evaluation_batches_fn)
        self.save_checkpoint()

    def evaluate(self, evaluation_batches_fn):
        self.run(self.metric_init_op)
        for batch in evaluation_batches_fn():
            self.run(self.metric_update_ops,
                     feed_dict=self.feed_dict(batch, is_training=False))
        step = self.run(self.global_step)
        results = self.run(self.metric_values)
        self.validation_log.record(step, results)
        LOG.info("step %5d:   %s", step, self.result_formatter.format_dict(results))
        LOG.info("step %5d:   eval/class_cost/ensemble=%f", step, self.ensemble_eval(evaluation_batches_fn))

    # begin temporal ensembling evaluation #

    def ensemble_eval(self, evaluation_batches_fn):
        if not hasattr(self, 'ensemble_eval_probs_result'):
            self.ensemble_eval_probs_result = self.build_ensemble_eval_probs(self.class_logits_1, self.labels)
         
        ensemble_eval_prediction_list = []
        for batch in evaluation_batches_fn():
            ensemble_eval_prediction_list.extend(list(self.run(self.ensemble_eval_probs_result, feed_dict=self.feed_dict(batch, is_training=False))))

        if not hasattr(self, 'ensemble_eval_result'):
            self.ensemble_eval_result = self.build_ensemble_eval(len(ensemble_eval_prediction_list), decay=self.ema_decay)

        return self.run(self.ensemble_eval_result, feed_dict={ self.ensemble_eval_predictions_placeholder : ensemble_eval_prediction_list })

    def build_ensemble_eval_probs(self, logits, labels):
        with tf.variable_scope("ensemble_eval_probs") as scope:
            applicable = tf.not_equal(labels, -1)

            # Change -1s to zeros to make cross-entropy computable
            labels = tf.where(applicable, labels, tf.zeros_like(labels))

            # This will now have incorrect values for unlabeled examples
            per_sample = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.class_logits_1, labels=labels)

            # Retain costs only for labeled
            per_sample = tf.where(applicable, per_sample, tf.zeros_like(per_sample))

            return tf.exp(-per_sample)

    def build_ensemble_eval(self, eval_sample_count, decay):
        with tf.variable_scope("ensemble_eval") as scope:
            self.ensemble_eval_ema = tf.train.ExponentialMovingAverage(decay=decay, zero_debias=True)
            self.ensemble_eval_predictions_placeholder = tf.placeholder(tf.float32, [eval_sample_count], name='predictions_placeholder') 
            self.ensemble_eval_predictions_var = tf.get_variable('predictions', [eval_sample_count], tf.float32, tf.constant_initializer(0), trainable=False)
            self.ensemble_eval_predictions = tf.identity(self.ensemble_eval_predictions_var)
            self.ensemble_eval_update_op = tf.assign(self.ensemble_eval_predictions_var, self.ensemble_eval_predictions_placeholder)
            with tf.control_dependencies([self.ensemble_eval_update_op]):
                self.ensemble_eval_ema_op = self.ensemble_eval_ema.apply([self.ensemble_eval_predictions])
            self.run(tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="ensemble_eval")))
            with tf.control_dependencies([self.ensemble_eval_ema_op]):
                return tf.reduce_mean(-tf.log(self.ensemble_eval_ema.average(self.ensemble_eval_predictions)))

    # end temporal ensembling evaluation #

    def get_training_control(self):
        return self.session.run(self.training_control)

    def run(self, *args, **kwargs):
        return self.session.run(*args, **kwargs)

    def feed_dict(self, batch, is_training=True, extra={}):
        result = dict({
            self.images: batch['x'],
            self.labels: batch['y'],
            self.is_training: is_training
        })
        result.update(extra)
        return result

    def save_checkpoint(self):
        path = self.saver.save(self.session, self.checkpoint_path, global_step=self.global_step)
        LOG.info("Saved checkpoint: %r", path)

    def save_tensorboard_graph(self):
        writer = tf.summary.FileWriter(self.tensorboard_path)
        writer.add_graph(self.session.graph)
        return writer.get_logdir()


Hyperparam = namedtuple("Hyperparam", ['tensor', 'getter', 'setter'])


def training_control(global_step, print_span, evaluation_span, max_step, name=None):
    with tf.name_scope(name, "training_control"):
        return {
            "step": global_step,
            "time_to_print": tf.equal(tf.mod(global_step, print_span), 0),
            "time_to_evaluate": tf.equal(tf.mod(global_step, evaluation_span), 0),
            "time_to_stop": tf.greater_equal(global_step, max_step),
        }


def step_rampup(global_step, rampup_length):
    result = tf.cond(global_step < rampup_length,
                     lambda: tf.constant(0.0),
                     lambda: tf.constant(1.0))
    return tf.identity(result, name="step_rampup")


def sigmoid_rampup(global_step, rampup_length):
    global_step = tf.to_float(global_step)
    rampup_length = tf.to_float(rampup_length)
    def ramp():
        phase = 1.0 - tf.maximum(0.0, global_step) / rampup_length
        return tf.exp(-5.0 * phase * phase)

    result = tf.cond(global_step < rampup_length, ramp, lambda: tf.constant(1.0))
    return tf.identity(result, name="sigmoid_rampup")


def sigmoid_rampdown(global_step, rampdown_length, training_length):
    global_step = tf.to_float(global_step)
    rampdown_length = tf.to_float(rampdown_length)
    training_length = tf.to_float(training_length)
    def ramp():
        phase = 1.0 - tf.maximum(0.0, training_length - global_step) / rampdown_length
        return tf.exp(-12.5 * phase * phase)

    result = tf.cond(global_step >= training_length - rampdown_length,
                     ramp,
                     lambda: tf.constant(1.0))
    return tf.identity(result, name="sigmoid_rampdown")


def inference(inputs, is_training, ema_decay, input_noise, student_dropout_probability, teacher_dropout_probability,
              normalize_input, flip_horizontally, translate, num_logits, layer_collection=None):
    tower_args = dict(inputs=inputs,
                      is_training=is_training,
                      input_noise=input_noise,
                      normalize_input=normalize_input,
                      flip_horizontally=flip_horizontally,
                      translate=translate,
                      num_logits=num_logits)

    with tf.variable_scope("initialization") as var_scope:
        init_pass = tower(**tower_args, dropout_probability=student_dropout_probability, is_initialization=True)
    with name_variable_scope("primary", var_scope, reuse=True) as (name_scope, _):
        class_logits_1, cons_logits_1 = tower(**tower_args, dropout_probability=student_dropout_probability, name=name_scope, layer_collection=layer_collection)
    with name_variable_scope("secondary", var_scope, reuse=True) as (name_scope, _):
        class_logits_2, cons_logits_2 = tower(**tower_args, dropout_probability=teacher_dropout_probability, name=name_scope)
    with ema_variable_scope("ema", var_scope, decay=ema_decay):
        class_logits_ema, cons_logits_ema = tower(**tower_args, dropout_probability=teacher_dropout_probability, name=name_scope)
        class_logits_ema, cons_logits_ema = tf.stop_gradient(class_logits_ema), tf.stop_gradient(cons_logits_ema)
    return init_pass, (class_logits_1, cons_logits_1), (class_logits_2, cons_logits_2), (class_logits_ema, cons_logits_ema)


def tower(inputs,
          is_training,
          dropout_probability,
          input_noise,
          normalize_input,
          flip_horizontally,
          translate,
          num_logits,
          layer_collection=None,
          is_initialization=False,
          name=None):
    with tf.name_scope(name, "tower"):
        default_conv_args = dict(
            padding='SAME',
            kernel_size=[3, 3],
            activation_fn=nn.lrelu,
            layer_collection=layer_collection,
            init=is_initialization
        )
        training_mode_funcs = [
            nn.random_translate, nn.flip_randomly, nn.gaussian_noise, slim.dropout,
            wn.fully_connected, wn.conv2d
        ]
        training_args = dict(
            is_training=is_training
        )

        with \
        slim.arg_scope([wn.conv2d], **default_conv_args), \
        slim.arg_scope(training_mode_funcs, **training_args):
            #pylint: disable=no-value-for-parameter
            net = inputs
            assert_shape(net, [None, 32, 32, 3])

            net = tf.cond(normalize_input,
                          lambda: slim.layer_norm(net,
                                                  scale=False,
                                                  center=False,
                                                  scope='normalize_inputs'),
                          lambda: net)
            assert_shape(net, [None, 32, 32, 3])

            net = nn.flip_randomly(net,
                                   horizontally=flip_horizontally,
                                   vertically=False,
                                   name='random_flip')
            net = tf.cond(translate,
                          lambda: nn.random_translate(net, scale=2, name='random_translate'),
                          lambda: net)
            net = nn.gaussian_noise(net, scale=input_noise, name='gaussian_noise')

            net = wn.conv2d(net, 128, scope="conv_1_1")
            net = wn.conv2d(net, 128, scope="conv_1_2")
            net = wn.conv2d(net, 128, scope="conv_1_3")
            net = slim.max_pool2d(net, [2, 2], scope='max_pool_1')
            net = slim.dropout(net, 1 - dropout_probability, scope='dropout_probability_1')
            assert_shape(net, [None, 16, 16, 128])

            net = wn.conv2d(net, 256, scope="conv_2_1")
            net = wn.conv2d(net, 256, scope="conv_2_2")
            net = wn.conv2d(net, 256, scope="conv_2_3")
            net = slim.max_pool2d(net, [2, 2], scope='max_pool_2')
            net = slim.dropout(net, 1 - dropout_probability, scope='dropout_probability_2')
            assert_shape(net, [None, 8, 8, 256])

            net = wn.conv2d(net, 512, padding='VALID', scope="conv_3_1")
            assert_shape(net, [None, 6, 6, 512])
            net = wn.conv2d(net, 256, kernel_size=[1, 1], scope="conv_3_2")
            net = wn.conv2d(net, 128, kernel_size=[1, 1], scope="conv_3_3")
            net = slim.avg_pool2d(net, [6, 6], scope='avg_pool')
            assert_shape(net, [None, 1, 1, 128])

            net = slim.flatten(net)
            assert_shape(net, [None, 128])

            primary_logits = wn.fully_connected(net, 10, init=is_initialization, layer_collection=layer_collection)
            secondary_logits = primary_logits#wn.fully_connected(net, 10, init=is_initialization)

            with tf.control_dependencies([tf.assert_greater_equal(num_logits, 1),
                                          tf.assert_less_equal(num_logits, 2)]):
                secondary_logits = tf.case([
                    (tf.equal(num_logits, 1), lambda: primary_logits),
                    (tf.equal(num_logits, 2), lambda: secondary_logits),
                ], exclusive=True, default=lambda: primary_logits)

            assert_shape(primary_logits, [None, 10])
            assert_shape(secondary_logits, [None, 10])
            return primary_logits, secondary_logits


def errors(logits, labels, name=None):
    """Compute error mean and whether each unlabeled example is erroneous

    Assume unlabeled examples have label == -1.
    Compute the mean error over unlabeled examples.
    Mean error is NaN if there are no unlabeled examples.
    Note that unlabeled examples are treated differently in cost calculation.
    """
    with tf.name_scope(name, "errors") as scope:
        applicable = tf.not_equal(labels, -1)
        labels = tf.boolean_mask(labels, applicable)
        logits = tf.boolean_mask(logits, applicable)
        predictions = tf.argmax(logits, -1)
        labels = tf.cast(labels, tf.int64)
        per_sample = tf.to_float(tf.not_equal(predictions, labels))
        mean = tf.reduce_mean(per_sample, name=scope)
        return mean, per_sample


def ensemble_classification_costs(logits, ensemble_probs, cons_coefficient, mask, name=None):
    num_classes = 10
    with tf.name_scope(name, "ensemble_classification_costs") as scope:
        full_mask = tf.logical_and(mask, tf.not_equal(tf.reduce_sum(ensemble_probs),0))
        target_probs = tf.where(full_mask, ensemble_probs, tf.zeros_like(ensemble_probs))

        #kl_cost_multiplier = 2 * (1 - 1/num_classes) / num_classes**2
        #costs = tf.nn.softmax_cross_entropy_with_logits(labels=target_probs, logits=logits) * kl_cost_multiplier
        costs = tf.reduce_mean((target_probs - tf.nn.softmax(logits)) ** 2, -1)

        # Take mean over all examples, not just labeled examples.
        costs = costs * tf.to_float(full_mask) * cons_coefficient
        labeled_sum = tf.reduce_sum(costs)
        total_count = tf.to_float(tf.shape(costs)[0])
        mean_cost = tf.div(labeled_sum, total_count, name=scope)

        assert_shape(costs, [None])
        assert_shape(mean_cost, [])

        return mean_cost, costs 

def classification_costs(logits, labels, name=None):
    """Compute classification cost mean and classification cost per sample

    Assume unlabeled examples have label == -1. For unlabeled examples, cost == 0.
    Compute the mean over all examples.
    Note that unlabeled examples are treated differently in error calculation.
    """
    with tf.name_scope(name, "classification_costs") as scope:
        applicable = tf.not_equal(labels, -1)

        # Change -1s to zeros to make cross-entropy computable
        labels = tf.where(applicable, labels, tf.zeros_like(labels))

        # This will now have incorrect values for unlabeled examples
        per_sample = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)

        # Retain costs only for labeled
        per_sample = tf.where(applicable, per_sample, tf.zeros_like(per_sample))

        # Take mean over all examples, not just labeled examples.
        labeled_sum = tf.reduce_sum(per_sample)
        total_count = tf.to_float(tf.shape(per_sample)[0])
        mean = tf.div(labeled_sum, total_count, name=scope)

        return mean, per_sample


def consistency_costs(logits1, logits2, cons_coefficient, mask, consistency_trust, name=None):
    """Takes a softmax of the logits and returns their distance as described below

    Consistency_trust determines the distance metric to use
    - trust=0: MSE
    - 0 < trust < 1: a scaled KL-divergence but both sides mixtured with
      a uniform distribution with given trust used as the mixture weight
    - trust=1: scaled KL-divergence

    When trust > 0, the cost is scaled to make the gradients
    the same size as MSE when trust -> 0. The scaling factor used is
    2 * (1 - 1/num_classes) / num_classes**2 / consistency_trust**2 .
    To have consistency match the strength of classification, use
    consistency coefficient = num_classes**2 / (1 - 1/num_classes) / 2
    which is 55.5555... when num_classes=10.

    Two potential stumbling blokcs:
    - When trust=0, this gives gradients to both logits, but when trust > 0
      this gives gradients only towards the first logit.
      So do not use trust > 0 with the Pi model.
    - Numerics may be unstable when 0 < trust < 1.
    """

    with tf.name_scope(name, "consistency_costs") as scope:
        num_classes = 10
        assert_shape(logits1, [None, num_classes])
        assert_shape(logits2, [None, num_classes])
        assert_shape(cons_coefficient, [])
        softmax1 = tf.nn.softmax(logits1)
        softmax2 = tf.nn.softmax(logits2)

        kl_cost_multiplier = 2 * (1 - 1/num_classes) / num_classes**2 / consistency_trust**2

        def pure_mse():
            costs = tf.reduce_mean((softmax1 - softmax2) ** 2, -1)
            return costs

        def pure_kl():
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits1, labels=softmax2)
            entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits2, labels=softmax2)
            costs = cross_entropy - entropy
            costs = costs * kl_cost_multiplier
            return costs

        def mixture_kl():
            with tf.control_dependencies([tf.assert_greater(consistency_trust, 0.0),
                                          tf.assert_less(consistency_trust, 1.0)]):
                uniform = tf.constant(1 / num_classes, shape=[num_classes])
                mixed_softmax1 = consistency_trust * softmax1 + (1 - consistency_trust) * uniform
                mixed_softmax2 = consistency_trust * softmax2 + (1 - consistency_trust) * uniform
                costs = tf.reduce_sum(mixed_softmax2 * tf.log(mixed_softmax2 / mixed_softmax1), axis=1)
                costs = costs * kl_cost_multiplier
                return costs

        costs = tf.case([
            (tf.equal(consistency_trust, 0.0), pure_mse),
            (tf.equal(consistency_trust, 1.0), pure_kl)
        ], default=mixture_kl)

        costs = costs * tf.to_float(mask) * cons_coefficient
        mean_cost = tf.reduce_mean(costs, name=scope)
        assert_shape(costs, [None])
        assert_shape(mean_cost, [])
        return mean_cost, costs


def total_costs(*all_costs, name=None):
    with tf.name_scope(name, "total_costs") as scope:
        for cost in all_costs:
            assert_shape(cost, [None])
        costs = tf.reduce_sum(all_costs)
        mean_cost = tf.reduce_mean(costs, name=scope)
        return mean_cost, costs
