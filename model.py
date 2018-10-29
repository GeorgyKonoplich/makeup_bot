from __future__ import division
from collections import namedtuple

from module import *
from utils import *


class PairedCycleGAN(object):
    def __init__(self, prefix, args):
        self.sess = None
        self.prefix = prefix
        self.image_size = args.fine_size
        self.input_c_dim = args.input_nc
        self.output_c_dim = args.output_nc

        self.discriminator = discriminator
        self.generator = generator_drnet
        self.generator_styled = generator_drnet_styled

        OPTIONS = namedtuple('OPTIONS', 'batch_size image_size \
                              gf_dim df_dim output_c_dim is_training')
        self.options = OPTIONS._make((args.batch_size, args.fine_size,
                                      args.ngf, args.ndf, args.output_nc,
                                      False))

        self._build_model()

    def _build_model(self):
        self.test_A = tf.placeholder(tf.float32,
                                     [None, self.image_size, self.image_size,
                                      self.input_c_dim], name='test_A')
        self.test_B = tf.placeholder(tf.float32,
                                     [None, self.image_size, self.image_size,
                                      self.output_c_dim], name='test_B')

        self.test_B_ = self.generator_styled(self.test_A, self.test_B, self.options, False, name=self.prefix+"generatorA2B")

    def test_bot(self, args, img_orig, b_path, intensity):
        out_var, in_var_a, in_var_b = (self.test_B_, self.test_A, self.test_B)

        sample_image_b = [load_test_data(b_path, args.fine_size)]
        sample_image = scipy.misc.imresize(img_orig, [args.fine_size, args.fine_size]) / 127.5 - 1.

        sample_image = np.array([sample_image]).astype(np.float32)
        sample_image_b = np.array(sample_image_b).astype(np.float32)
        fake_img, fake_delta = self.sess.run(out_var, feed_dict={in_var_a: sample_image, in_var_b: sample_image_b})
        fake_delta = fake_delta[0]
        edge = img_orig.shape[:2]
        fake_delta_resize = scipy.misc.imresize((inverse_transform(fake_delta) * 255).astype(np.uint8),
                                                [edge[1], edge[0]])
        new_image = np.clip((fake_delta_resize / 127.5 - 1) * (intensity / 100.0) + (img_orig / 127.5 - 1), -1, 1)
        return (inverse_transform(new_image) * 255).astype(np.uint8)


class CycleGAN(object):
    def __init__(self, prefix, args):
        self.sess = None
        self.prefix = prefix
        self.image_size = args.fine_size
        self.input_c_dim = args.input_nc
        self.output_c_dim = args.output_nc

        self.discriminator = discriminator
        self.generator = generator_drnet

        OPTIONS = namedtuple('OPTIONS', 'batch_size image_size \
                              gf_dim df_dim output_c_dim is_training')
        self.options = OPTIONS._make((args.batch_size, args.fine_size,
                                      args.ngf, args.ndf, args.output_nc,
                                      False))

        self._build_model()

    def _build_model(self):
        self.test_A = tf.placeholder(tf.float32,
                                     [None, self.image_size, self.image_size,
                                      self.input_c_dim], name='test_A')
        self.test_B = tf.placeholder(tf.float32,
                                     [None, self.image_size, self.image_size,
                                      self.output_c_dim], name='test_B')

        self.test_B_ = self.generator(self.test_A, self.options, False, name=self.prefix+"generatorA2B")

    def test_bot(self, args, img_orig, b_path, intensity):
        out_var, in_var_a, in_var_b = (self.test_B_, self.test_A, self.test_B)

        sample_image_b = [load_test_data(b_path, args.fine_size)]
        sample_image = scipy.misc.imresize(img_orig, [args.fine_size, args.fine_size]) / 127.5 - 1.

        sample_image = np.array([sample_image]).astype(np.float32)
        sample_image_b = np.array(sample_image_b).astype(np.float32)
        fake_img, fake_delta = self.sess.run(out_var, feed_dict={in_var_a: sample_image, in_var_b: sample_image_b})
        fake_delta = fake_delta[0]
        edge = img_orig.shape[:2]
        fake_delta_resize = scipy.misc.imresize((inverse_transform(fake_delta) * 255).astype(np.uint8),
                                                [edge[1], edge[0]])
        new_image = np.clip((fake_delta_resize / 127.5 - 1) * (intensity / 100.0) + (img_orig / 127.5 - 1), -1, 1)
        return (inverse_transform(new_image) * 255).astype(np.uint8)
