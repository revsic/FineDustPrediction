import os
import model
import preprocess
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_float('learning_rate', 1e-4, 'Float, learning rate, default 1e-4.')
flags.DEFINE_float('beta1', 0.9, 'Float, beta1 value in Adam, default 0.9.')
flags.DEFINE_integer('epoch', 100, 'Integer, number of epochs, default 20.')
flags.DEFINE_integer('batch_size', 55, 'Integer, size of batch, default 128.')
flags.DEFINE_integer('ckpt_interval', 5, 'Integer, interval for writing checkpoint, default 5')
flags.DEFINE_string('name', 'default', 'String, name of model, default `default`.')
flags.DEFINE_string('summary_dir', 'summary', 'String, dir name for saving tensor summary, default `./summary`.')
flags.DEFINE_string('ckpt_dir', 'ckpt', 'String, dir name for saving checkpoint, default `./ckpt_dir`.')
FLAGS = flags.FLAGS


def main(_):
    # total_x, total_y, x_dim, y_dim
    ckpt_path = os.path.join(FLAGS.ckpt_dir, FLAGS.name)
    (train_x, train_y), (test_x, test_y) = preprocess.create_dataset()

    batch = model.Batch(train_x, train_y, FLAGS.epoch)

    print('start session')
    with tf.Session() as sess:
        predicator = model.Predicator(matrix_shape=[9, 8],
                                      num_time=7,
                                      out_time=7,
                                      kernels=[[5, 5], [5, 5], [5, 5], [5, 5], [5, 5]],
                                      depths=[256, 128, 128, 64, 32],
                                      learning_rate=FLAGS.learning_rate,
                                      beta1=FLAGS.beta1)

        train_path = os.path.join(FLAGS.summary_dir, FLAGS.name, 'train')
        test_path = os.path.join(FLAGS.summary_dir, FLAGS.name, 'test')

        train_writer = tf.summary.FileWriter(train_path, sess.graph)
        test_writer = tf.summary.FileWriter(test_path, sess.graph)

        print('start training')
        sess.run(tf.global_variables_initializer())
        for i in range(FLAGS.epoch):
            for n in range(batch.iter_per_epoch):
                batch_x, batch_y = batch()
                predicator.train(sess, batch_x, batch_y)

            print(i, 'th epoch')
            summary = predicator.inference(sess, predicator.summary, batch_x, batch_y)
            train_writer.add_summary(summary, global_step=i)

            summary = predicator.inference(sess, predicator.summary, test_x, test_y)
            test_writer.add_summary(summary, global_step=i)

            if (i + 1) % FLAGS.ckpt_interval == 0:
                predicator.dump(sess, ckpt_path, i)


if __name__ == '__main__':
    tf.app.run()
