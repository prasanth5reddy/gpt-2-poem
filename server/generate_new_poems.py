import json
import os
import re
import warnings

import numpy as np
import tensorflow as tf

from server.poems_table import Poems

warnings.filterwarnings("ignore")
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import src.model as model, src.sample as sample, src.encoder as encoder


def sample_model(
        model_name='117M',
        use_seed=False,
        seed=None,
        nsamples=0,
        batch_size=1,
        length=None,
        temperature=1.0,
        top_k=0,
        top_p=0.0,
):
    """
    Run the sample_model
    :model_name=117M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to
     reproduce results
    :nsamples=0 : Number of samples to return, if 0, continues to
     generate samples indefinately.
    :batch_size=1 : Number of batches (only affects speed/memory).
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
    :top_p=0.0 : Float value controlling diversity. Implements nucleus sampling,
     overriding top_k if set to a value > 0. A good setting is 0.9.
    """
    poems = []
    enc = encoder.get_encoder(model_name)
    hparams = model.default_hparams()
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    with tf.Session(graph=tf.Graph()) as sess:
        if use_seed:
            np.random.seed(seed)
            tf.set_random_seed(seed)

        output = sample.sample_sequence(
            hparams=hparams, length=length,
            start_token=enc.encoder['<|endoftext|>'],
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p
        )[:, 1:]

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name))
        saver.restore(sess, ckpt)

        generated = 0
        while nsamples == 0 or generated < nsamples:
            out = sess.run(output)
            for i in range(batch_size):
                generated += batch_size
                text = enc.decode(out[i])
                # print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                text = re.sub("[0-9]+.|", "", text)
                print(text)
                poems.append(text.encode('utf-8'))
    return poems


def main():
    model_name = '345M-poetry'
    top_k = 40
    temperature = 0.9
    nsamples = 500
    poems_gen = sample_model(model_name=model_name, top_k=top_k, temperature=temperature, nsamples=nsamples)
    poems_table = Poems()
    poems_table.write_to_table(poems_gen)


if __name__ == '__main__':
    main()
