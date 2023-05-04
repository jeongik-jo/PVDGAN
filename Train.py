import tensorflow as tf
from tensorflow import keras as kr
import HyperParameters as hp


@tf.function
def _dlsgan_train_step(encoder: kr.Model, decoder: kr.Model, latent_var_trace: tf.Variable, real_images: tf.Tensor):
    with tf.GradientTape(persistent=True) as tape:
        real_images = tf.image.random_flip_left_right(real_images)
        batch_size = real_images.shape[0]
        latent_dim = tf.cast(hp.latent_dim, 'float32')
        latent_scale_vector = tf.sqrt(latent_dim * latent_var_trace / tf.reduce_sum(latent_var_trace))

        with tf.GradientTape() as reg_tape:
            reg_tape.watch(real_images)
            real_adv_values, real_latent_vectors, _, real_feature_vectors = encoder(real_images)
        reg_loss = tf.reduce_mean(tf.reduce_sum(tf.square(reg_tape.gradient(real_adv_values, real_images)), axis=[1, 2, 3]))

        if hp.use_image_rec:
            half_batch_size = batch_size // 2
            latent_vectors = tf.concat([hp.latent_dist_func(half_batch_size),
                                        tf.stop_gradient(real_latent_vectors[half_batch_size:]) +
                                        hp.latent_dist_func(half_batch_size) * tf.sqrt(tf.nn.relu(1 - latent_var_trace))[tf.newaxis]], axis=0)
            fake_images = decoder(latent_vectors * latent_scale_vector[tf.newaxis])
            fake_adv_values, rec_latent_vectors, _, rec_feature_vectors = encoder(fake_images)
            image_rec_loss = tf.reduce_mean(tf.square(rec_feature_vectors[half_batch_size:] - real_feature_vectors[half_batch_size:]))
        else:
            latent_vectors = hp.latent_dist_func(batch_size)
            fake_images = decoder(latent_vectors * latent_scale_vector[tf.newaxis])
            fake_adv_values, rec_latent_vectors, _, _ = encoder(fake_images)
            image_rec_loss = 0.0
        latent_rec_loss = tf.reduce_mean(tf.square((latent_vectors - rec_latent_vectors) * latent_scale_vector[tf.newaxis]))

        dis_adv_loss = tf.reduce_mean(tf.nn.softplus(-real_adv_values) + tf.nn.softplus(fake_adv_values))
        gen_adv_loss = tf.reduce_mean(tf.nn.softplus(-fake_adv_values))

        enc_loss = dis_adv_loss + hp.latent_rec_weight * latent_rec_loss + hp.reg_weight * reg_loss
        dec_loss = gen_adv_loss + hp.latent_rec_weight * latent_rec_loss + hp.image_rec_weight * image_rec_loss

    hp.enc_opt.apply_gradients(
        zip(tape.gradient(enc_loss, encoder.trainable_variables),
            encoder.trainable_variables)
    )
    hp.dec_opt.apply_gradients(
        zip(tape.gradient(dec_loss, decoder.trainable_variables),
            decoder.trainable_variables)
    )

    hp.enc_ema.apply(encoder.trainable_variables)
    hp.dec_ema.apply(decoder.trainable_variables)
    latent_var_trace.assign(latent_var_trace * hp.latent_var_decay_rate +
                            tf.reduce_mean(tf.square(rec_latent_vectors), axis=0) * (1.0 - hp.latent_var_decay_rate))

    results = {
        'real_adv_values': real_adv_values, 'fake_adv_values': fake_adv_values,
        'reg_loss': reg_loss, 'latent_rec_loss': latent_rec_loss, 'image_rec_loss': image_rec_loss,
    }
    return results

@tf.function
def _vaegan_train_step(encoder: kr.Model, decoder: kr.Model, discriminator: kr.Model, real_images: tf.Tensor):
    with tf.GradientTape(persistent=True) as tape:
        real_images = tf.image.random_flip_left_right(real_images)
        batch_size = real_images.shape[0]
        half_batch_size = batch_size // 2

        with tf.GradientTape() as reg_tape:
            reg_tape.watch(real_images)
            real_adv_values, _, _, real_feature_vectors = discriminator(real_images)
        reg_loss = tf.reduce_mean(tf.reduce_sum(tf.square(reg_tape.gradient(real_adv_values, real_images)), axis=[1, 2, 3]))

        _, latent_means, latent_log_vars, _ = encoder(real_images[half_batch_size:])
        latent_vectors = tf.concat([hp.latent_dist_func(half_batch_size),
                                    hp.latent_dist_func(half_batch_size) * tf.exp(latent_log_vars / 2) + latent_means], axis=0)
        fake_images = decoder(latent_vectors)
        fake_adv_values, _, _, rec_feature_vectors = discriminator(fake_images)
        image_rec_loss = tf.reduce_mean(tf.square(rec_feature_vectors[half_batch_size:] - real_feature_vectors[half_batch_size:]))
        prior_loss = tf.reduce_mean(tf.square(latent_means) - latent_log_vars + tf.exp(latent_log_vars))

        dis_adv_loss = tf.reduce_mean(tf.nn.softplus(-real_adv_values) + tf.nn.softplus(fake_adv_values))
        gen_adv_loss = tf.reduce_mean(tf.nn.softplus(-fake_adv_values))

        enc_loss = hp.image_rec_weight * image_rec_loss + hp.prior_weight * prior_loss
        dec_loss = gen_adv_loss + hp.image_rec_weight * image_rec_loss
        dis_loss = dis_adv_loss + hp.reg_weight * reg_loss

    hp.enc_opt.apply_gradients(
        zip(tape.gradient(enc_loss, encoder.trainable_variables),
            encoder.trainable_variables)
    )
    hp.dec_opt.apply_gradients(
        zip(tape.gradient(dec_loss, decoder.trainable_variables),
            decoder.trainable_variables)
    )
    hp.dis_opt.apply_gradients(
        zip(tape.gradient(dis_loss, discriminator.trainable_variables),
            discriminator.trainable_variables)
    )

    hp.enc_ema.apply(encoder.trainable_variables)
    hp.dec_ema.apply(decoder.trainable_variables)

    results = {
        'real_adv_values': real_adv_values, 'fake_adv_values': fake_adv_values,
        'reg_loss': reg_loss, 'prior_loss': prior_loss, 'image_rec_loss': image_rec_loss,
    }
    return results


def train(encoder: kr.Model, decoder: kr.Model, discriminator: kr.Model, latent_var_trace: tf.Variable, dataset):
    results = {}
    for data in dataset:
        if hp.is_dls:
            batch_results = _dlsgan_train_step(encoder, decoder, latent_var_trace, data)
        else:
            batch_results = _vaegan_train_step(encoder, decoder, discriminator, data)
        for key in batch_results:
            try:
                results[key].append(batch_results[key])
            except KeyError:
                results[key] = [batch_results[key]]

    temp_results = {}
    for key in results:
        mean, variance = tf.nn.moments(tf.concat(results[key], axis=0), axes=0)
        temp_results[key + '_mean'] = mean
        temp_results[key + '_variance'] = variance
    results = temp_results

    for key in results:
        print('%-30s:' % key, '%13.6f' % results[key].numpy())

    return results
