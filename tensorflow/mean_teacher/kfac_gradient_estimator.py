from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.kfac.python.ops import estimator as fisher_est
from tensorflow.contrib.kfac.python.ops import utils as fisher_utils
from tensorflow.contrib.kfac.python.ops import fisher_factors

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import gradient_descent
        
import types 

# LayerCollection
def lc_check_registration(self, variables):
    pass

# FisherFactor
def ff_register_damped_inverse(self, damping):
    if not hasattr(self, "_sqrts_by_damping"):
       self._sqrts_by_damping = {}
    if damping not in self._sqrts_by_damping:
        damping_string = fisher_factors.scalar_or_tensor_to_string(damping)
        with variable_scope.variable_scope(self._var_scope):
            sqrt = variable_scope.get_variable(
                "sqrt_damp{}".format(damping_string),
                initializer=fisher_factors.inverse_initializer,
                shape=self._cov_shape,
                trainable=False,
                dtype=self._dtype)
            self._sqrts_by_damping[damping] = sqrt 

# FisherFactor
def ff_make_inverse_update_ops(self):
    ops = self.original_make_inverse_update_ops()
    self.register_eigendecomp()
    eigenvalues, eigenvectors = self._eigendecomp
    clipped_eigenvalues = math_ops.maximum(eigenvalues,
                                           fisher_factors.EIGENVALUE_CLIPPING_THRESHOLD)
    for damping, sqrt in self._sqrts_by_damping.items():
        ops.append(sqrt.assign(eigenvectors / math_ops.sqrt(clipped_eigenvalues + damping))) 
    return ops

# FisherBlock
def fb_instantiate_factors(self, grads_list, damping):
    self.original_instantiate_factors(grads_list, damping)

    ff_register_damped_inverse(self._input_factor, self._input_damping)
    ff_register_damped_inverse(self._output_factor, self._output_damping)

    for fisher_factor in [self._input_factor, self._output_factor]:
        fisher_factor.original_make_inverse_update_ops = fisher_factor.make_inverse_update_ops 
        fisher_factor.make_inverse_update_ops = types.MethodType(ff_make_inverse_update_ops, fisher_factor)

# FisherBlock
def fb_scale_and_add_noise(self, vector, main_scale, noise_scale):
    left_sqrt = self._input_factor._sqrts_by_damping[self._input_damping]
    right_sqrt = self._output_factor._sqrts_by_damping[self._output_damping]
    
    reshaped_vector = fisher_utils.layer_params_to_mat2d(vector)
    noise_vector = tf.random_normal(shape=tf.shape(reshaped_vector))
    reshaped_out = reshaped_vector + \
                   math_ops.sqrt(noise_scale / self._renorm_coeff) / main_scale * \
                   math_ops.matmul(left_sqrt,math_ops.matmul(noise_vector,array_ops.transpose(right_sqrt)))
    return fisher_utils.mat2d_to_layer_params(vector, reshaped_out)

class NaturalGradientOptimizer(gradient_descent.GradientDescentOptimizer):
  def __init__(self,
               main_learning_rate,
               noise_learning_scale,
               cov_ema_decay,
               damping,
               layer_collection,
               var_list=None,
               momentum=0.0,
               sampling_type='sgld',
               norm_constraint=None,
               name="KFAC",
               estimation_mode="gradients",
               colocate_gradients_with_ops=True,
               cov_devices=None,
               inv_devices=None):
        variables = var_list
        if variables is None:
          variables = tf_variables.trainable_variables()

        # begin monkey patching

        layer_collection.check_registration = types.MethodType(lc_check_registration, layer_collection)

        for fisher_block in layer_collection.get_blocks():
            fisher_block.scale_and_add_noise = types.MethodType(fb_scale_and_add_noise, fisher_block) 
            fisher_block.original_instantiate_factors = fisher_block.instantiate_factors
            fisher_block.instantiate_factors = types.MethodType(fb_instantiate_factors, fisher_block)

        # end monkey patching

        self._fisher_est = fisher_est.FisherEstimator(
            variables,
            cov_ema_decay,
            damping,
            layer_collection,
            estimation_mode=estimation_mode,
            colocate_gradients_with_ops=colocate_gradients_with_ops,
            cov_devices=cov_devices,
            inv_devices=inv_devices)
       
        self._main_learning_rate = main_learning_rate
        self._noise_learning_scale = noise_learning_scale
        self._momentum = momentum
        self._sampling_type = sampling_type
        self._norm_constraint = norm_constraint

        self._batch_size = array_ops.shape(layer_collection.losses[0].inputs)[0]
        self._losses = layer_collection.losses

        super(NaturalGradientOptimizer, self).__init__(learning_rate=main_learning_rate, name=name)

  @property
  def cov_update_thunks(self):
    return self._fisher_est.cov_update_thunks

  @property
  def cov_update_ops(self):
    return self._fisher_est.cov_update_ops

  @property
  def cov_update_op(self):
    return self._fisher_est.cov_update_op

  @property
  def inv_update_thunks(self):
    return self._fisher_est.inv_update_thunks

  @property
  def inv_update_ops(self):
    return self._fisher_est.inv_update_ops

  @property
  def inv_update_op(self):
    return self._fisher_est.inv_update_op

  @property
  def variables(self):
    return self._fisher_est.variables

  @property
  def damping(self):
    return self._fisher_est.damping

  def minimize(self, *args, **kwargs):
    kwargs["var_list"] = kwargs.get("var_list") or self.variables
    if set(kwargs["var_list"]) != set(self.variables):
      raise ValueError("var_list doesn't match with set of Fisher-estimating "
                       "variables.")
    return super(NaturalGradientOptimizer, self).minimize(*args, **kwargs)

  def compute_gradients(self, *args, **kwargs):
    # args[1] could be our var_list
    if len(args) > 1:
      var_list = args[1]
    else:
      kwargs["var_list"] = kwargs.get("var_list") or self.variables
      var_list = kwargs["var_list"]
    if set(var_list) != set(self.variables):
      raise ValueError("var_list doesn't match with set of Fisher-estimating "
                       "variables.")
    return super(NaturalGradientOptimizer, self).compute_gradients(*args, **kwargs)

  def apply_gradients(self, grads_and_vars, *args, **kwargs):
    """Applies gradients to variables.
    Args:
      grads_and_vars: List of (gradient, variable) pairs.
      *args: Additional arguments for super.apply_gradients.
      **kwargs: Additional keyword arguments for super.apply_gradients.
    Returns:
      An `Operation` that applies the specified gradients.
    """
    # In Python 3, grads_and_vars can be a zip() object which can only be
    # iterated over once. By converting it to a list, we ensure that it can be
    # iterated over more than once.
    grads_and_vars = list(grads_and_vars)

    # Compute step.
    steps_and_vars = self._compute_update_steps(grads_and_vars)

    # Update trainable variables with this step.
    return super(NaturalGradientOptimizer, self).apply_gradients(steps_and_vars, *args,
                                                      **kwargs)               

  def _compute_update_steps(self, grads_and_vars):
    if self._sampling_type == 'gradient':
        precon_grads_and_vars = grads_and_vars
    else:
        precon_grads_and_vars = self._fisher_est._apply_transformation(grads_and_vars, lambda fb, vec: fb.multiply_inverse(vec))
        if self._sampling_type == 'natural_gradient' or self._sampling_type is None:
            pass
        elif self._sampling_type == 'sgld':
            precon_grads_and_vars = self._fisher_est._apply_transformation(precon_grads_and_vars, lambda fb, vec: fb.scale_and_add_noise(vec, math_ops.sqrt(self._main_learning_rate), 2*self._noise_learning_scale))
        elif self._sampling_type == 'sghmc':
            precon_grads_and_vars = self._fisher_est._apply_transformation(precon_grads_and_vars, lambda fb, vec: fb.scale_and_add_noise(vec, 1.0, self._noise_learning_scale))
        else:
            assert False 

    # Apply "KL clipping" if asked for.
    if self._norm_constraint is not None:
        precon_grads_and_vars = self._clip_updates(grads_and_vars,
                                                   precon_grads_and_vars)
 
    # Update the velocity with this and return it as the step.
    if self._momentum != 0:
        precon_grads_and_vars = self._update_velocities(precon_grads_and_vars, self._momentum)

    return precon_grads_and_vars 

  def _update_velocities(self, vecs_and_vars, decay, vec_coeff=1.0):
    def _update_velocity(vec, var):
      velocity = self._zeros_slot(var, "velocity", self._name)
      with ops.colocate_with(velocity):
        # NOTE(mattjj): read/modify/write race condition not suitable for async.

        # Compute the new velocity for this variable.
        new_velocity = decay * velocity + vec_coeff * vec

        # Save the updated velocity.
        return (array_ops.identity(velocity.assign(new_velocity)), var)

    # Go through variable and update its associated part of the velocity vector.
    return [_update_velocity(vec, var) for vec, var in vecs_and_vars]

  def _clip_updates(self, grads_and_vars, precon_grads_and_vars):
    coeff = self._update_clip_coeff(grads_and_vars, precon_grads_and_vars)
    return [(pgrad * coeff, var) for pgrad, var in precon_grads_and_vars]

  def _update_clip_coeff(self, grads_and_vars, precon_grads_and_vars):
    sq_norm_grad = self._squared_fisher_norm(grads_and_vars,
                                             precon_grads_and_vars)
    sq_norm_up = sq_norm_grad * self._learning_rate**2
    return math_ops.minimum(1.,
                            math_ops.sqrt(self._norm_constraint / sq_norm_up))

  def _squared_fisher_norm(self, grads_and_vars, precon_grads_and_vars):
    for (_, gvar), (_, pgvar) in zip(grads_and_vars, precon_grads_and_vars):
      if gvar is not pgvar:
        raise ValueError("The variables referenced by the two arguments "
                         "must match.")
    terms = [
        math_ops.reduce_sum(grad * pgrad)
        for (grad, _), (pgrad, _) in zip(grads_and_vars, precon_grads_and_vars)
    ]
    return math_ops.reduce_sum(terms)
