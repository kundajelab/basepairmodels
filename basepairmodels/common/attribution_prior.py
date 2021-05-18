from tensorflow.keras import Model

class AttributionPriorModel(Model):

    def __init__(self, frequency_limit, limit_softness, grad_smooth_sigma, 
                 **kwargs):
        super(CustomModel, self).__init__(**kwargs)
        
        self.freq_limit = frequency_limit
        self.limit_softness = limit_softness
        self.grad_smooth_sigma = grad_smooth_sigma
 
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            with tf.GradientTape() as input_grad_tape:
                input_grad_tape.watch(x)
                
                y_pred = self(x, training=True)  # Forward pass
                # Compute the loss value
                # (the loss function is configured in `compile()`)
                batch_loss = self.compiled_loss(
                    y, y_pred, regularization_losses=self.losses)

            if self.use_prior:
                
                # mean-normalize the profile logits output & weight by
                # post-softmax probabilities
                y_pred_profile = y_pred[0] - kb.mean(y_pred[0])
                y_pred_profile *= log_softmax(y_pred_profile)
                
                # Compute gradients of the output with respect to the input
                
                # gradients of profile output w.r.t to input
                input_grads_profile = input_grad_tape.gradient(
                    y_pred_profile, x['sequence'])
                # gradients of counts output w.r.t to input
                input_grads_counts = input_grad_tape.gradient(
                    y_pred[1], x['sequence'])
                
                # Gradient * input
                input_grads_profile = input_grads_profile * x['sequence']  
                input_grads_counts = input_grads_counts * x['sequence']  
                
                # attribution prior loss of profile
                batch_attr_prior_loss_profile = self.fourier_att_prior_loss(
                    x['sequence'], input_grads_profile, self.freq_limit, 
                    self.limit_softness,
                    self.att_prior_grad_smooth_sigma
                )

                # attribution prior loss of counts
                batch_attr_prior_loss_counts = self.fourier_att_prior_loss(
                    x['sequence'], input_grads_counts, self.freq_limit, 
                    self.limit_softness,
                    self.att_prior_grad_smooth_sigma
                )

                batch_loss = batch_loss + batch_attr_prior_loss_profile + \
                    batch_attr_prior_loss_counts

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(batch_loss, trainable_vars)
        
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        
        # Return a dict mapping metric names to current value
        metrics = {m.name: m.result() for m in self.metrics}
                
        if self.use_prior:
            metrics['batch_attribution_prior_loss'] = attribution_prior_loss
        
        return metrics
