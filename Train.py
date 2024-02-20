import tensorflow as tf
from tensorflow import keras as kr
import HyperParameters as hp


@tf.function
def _dlsgan_train_step(enc: kr.Model, dec: kr.Model, real_imgs: tf.Tensor):
    ltn_scl_vecs = hp.get_ltn_scl_vecs()
    batch_size = real_imgs.shape[0]
    half_batch_size = batch_size // 2

    ltn_vecs = hp.ltn_dist_func(batch_size)
    fake_imgs = dec(ltn_vecs * ltn_scl_vecs)

    with tf.GradientTape() as enc_tape:
        with tf.GradientTape() as reg_tape:
            reg_tape.watch(real_imgs)
            real_adv_vals, real_ltn_vecs, _, real_ftr_vecs = enc(real_imgs)
        reg_loss = tf.reduce_mean(tf.reduce_sum(tf.square(reg_tape.gradient(real_adv_vals, real_imgs)), axis=[1, 2, 3]))
        fake_adv_vals, rec_ltn_vecs, rec_ltn_logvars, _ = enc(fake_imgs)

        ltn_rec_diff = tf.square((ltn_vecs - rec_ltn_vecs) * ltn_scl_vecs)
        rec_ltn_traces = rec_ltn_vecs

        if hp.use_logvar:
            ltn_rec_loss = tf.reduce_mean(rec_ltn_logvars + ltn_rec_diff / (tf.exp(rec_ltn_logvars) + 1e-7))
        else:
            ltn_rec_loss = tf.reduce_mean(ltn_rec_diff)

        dis_adv_loss = tf.reduce_mean(tf.nn.softplus(-real_adv_vals) + tf.nn.softplus(fake_adv_vals))
        enc_loss = dis_adv_loss + hp.ltn_rec_w * ltn_rec_loss + hp.reg_w * reg_loss

    hp.enc_opt.minimize(enc_loss, enc.trainable_variables, tape=enc_tape)

    if hp.is_pvd:
        ltn_vecs = tf.concat([
            real_ltn_vecs[:half_batch_size] + hp.ltn_dist_func(half_batch_size) * tf.sqrt(tf.nn.relu(1 - hp.ltn_var_trace))[tf.newaxis],
            hp.ltn_dist_func(half_batch_size)
        ], axis=0)

    else:
        ltn_vecs = hp.ltn_dist_func(batch_size)

    with tf.GradientTape() as dec_tape:
        fake_imgs = dec(ltn_vecs * ltn_scl_vecs)
        fake_adv_vals, rec_ltn_vecs, rec_ltn_logvars, rec_ftr_vecs = enc(fake_imgs)

        if hp.is_pvd:
            img_rec_loss = tf.reduce_mean(tf.square((rec_ftr_vecs - real_ftr_vecs)[:half_batch_size]))
            ltn_rec_diff = tf.square((ltn_vecs - rec_ltn_vecs)[half_batch_size:] * ltn_scl_vecs)
            rec_ltn_logvars = rec_ltn_logvars[half_batch_size:]
            rec_ltn_traces = tf.concat([rec_ltn_traces, rec_ltn_vecs[half_batch_size:]], axis=0)
        else:
            img_rec_loss = tf.constant(0.0)
            ltn_rec_diff = tf.square((ltn_vecs - rec_ltn_vecs) * ltn_scl_vecs)
            rec_ltn_traces = tf.concat([rec_ltn_traces, rec_ltn_vecs], axis=0)

        if hp.use_logvar:
            ltn_rec_loss = tf.reduce_mean(rec_ltn_logvars + ltn_rec_diff / (tf.exp(rec_ltn_logvars) + 1e-7))
        else:
            ltn_rec_loss = tf.reduce_mean(ltn_rec_diff)

        gen_adv_loss = tf.reduce_mean(tf.nn.softplus(-fake_adv_vals))
        dec_loss = gen_adv_loss + hp.ltn_rec_w * ltn_rec_loss + hp.img_rec_w * img_rec_loss

    hp.dec_opt.minimize(dec_loss, dec.trainable_variables, tape=dec_tape)

    hp.ltn_var_trace.assign(hp.ltn_var_trace * hp.ltn_var_decay_rate +
                            tf.reduce_mean(tf.square(rec_ltn_traces), axis=0) * (1.0 - hp.ltn_var_decay_rate))

    results = {
        'real_adv_val': tf.reduce_mean(real_adv_vals), 'fake_adv_val': tf.reduce_mean(fake_adv_vals),
        'reg_loss': reg_loss, 'ltn_rec_loss': ltn_rec_loss, 'img_rec_loss': img_rec_loss,
    }
    return results


@tf.function
def _vaegan_train_step(enc: kr.Model, dec: kr.Model, dis: kr.Model, real_imgs: tf.Tensor):
    batch_size = real_imgs.shape[0]
    half_batch_size = batch_size // 2

    _, ltn_means, ltn_logvars, _ = enc(real_imgs[half_batch_size:])
    ltn_vecs = tf.concat([
        hp.ltn_dist_func(half_batch_size),
        hp.ltn_dist_func(half_batch_size) * tf.exp(ltn_logvars / 2) + ltn_means
    ], axis=0)
    fake_imgs = dec(ltn_vecs)

    with tf.GradientTape() as dis_tape:
        with tf.GradientTape() as reg_tape:
            reg_tape.watch(real_imgs)
            real_adv_vals, _, _, real_ftr_vecs = dis(real_imgs)
        reg_loss = tf.reduce_mean(tf.reduce_sum(tf.square(reg_tape.gradient(real_adv_vals, real_imgs)), axis=[1, 2, 3]))
        fake_adv_vals, _, _, _ = dis(fake_imgs)
        dis_adv_loss = tf.reduce_mean(tf.nn.softplus(-real_adv_vals) + tf.nn.softplus(fake_adv_vals))
        dis_loss = dis_adv_loss + hp.reg_w * reg_loss

    hp.dis_opt.minimize(dis_loss, dis.trainable_variables, tape=dis_tape)

    with tf.GradientTape(persistent=True) as gen_tape:
        _, ltn_means, ltn_logvars, _ = enc(real_imgs[:half_batch_size])
        prr_loss = tf.reduce_mean(tf.square(ltn_means) - ltn_logvars + tf.exp(ltn_logvars))

        ltn_vecs = tf.concat([
            hp.ltn_dist_func(half_batch_size) * tf.exp(ltn_logvars / 2) + ltn_means,
            hp.ltn_dist_func(half_batch_size)
        ], axis=0)

        fake_imgs = dec(ltn_vecs)
        fake_adv_vals, _, _, rec_ftr_vecs = dis(fake_imgs)

        img_rec_loss = tf.reduce_mean(tf.square((rec_ftr_vecs - real_ftr_vecs)[:half_batch_size]))
        gen_adv_loss = tf.reduce_mean(tf.nn.softplus(-fake_adv_vals))

        enc_loss = hp.img_rec_w * img_rec_loss + hp.prr_w * prr_loss
        dec_loss = gen_adv_loss + hp.img_rec_w * img_rec_loss

    hp.enc_opt.minimize(enc_loss, enc.trainable_variables, tape=gen_tape)
    hp.dec_opt.minimize(dec_loss, dec.trainable_variables, tape=gen_tape)

    results = {
        'real_adv_val': tf.reduce_mean(real_adv_vals), 'fake_adv_val': tf.reduce_mean(fake_adv_vals),
        'reg_loss': reg_loss, 'prr_loss': prr_loss, 'img_rec_loss': img_rec_loss,
    }
    return results


def train(enc: kr.Model, dec: kr.Model, dis: kr.Model, dataset):
    results = {}
    for data in dataset:
        if hp.is_dls:
            batch_results = _dlsgan_train_step(enc, dec, data)
        else:
            batch_results = _vaegan_train_step(enc, dec, dis, data)
        for key in batch_results:
            try:
                results[key].append(batch_results[key])
            except KeyError:
                results[key] = [batch_results[key]]

    temp_results = {}
    for key in results:
        mean, variance = tf.nn.moments(tf.convert_to_tensor(results[key]), axes=0)
        temp_results[key + '_mean'] = mean
        temp_results[key + '_variance'] = variance
    temp_results['ltn_ent'] = hp.get_ltn_ent()
    results = temp_results

    for key in results:
        print('%-30s:' % key, '%13.6f' % results[key].numpy())

    return results

