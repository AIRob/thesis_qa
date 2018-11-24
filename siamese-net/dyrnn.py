import tensorflow as tf
# sequence : [batch_size*max_seq_len] with zero paddings
# return : [batch_size] : value of seq_length (not contains zero paddings)
def seq_length(sequence):
    used = tf.sign(tf.abs(sequence))
    length = tf.reduce_sum(used, 1)
    length = tf.cast(length, tf.int32)
    return length

def last_relevant(output, seq_length):
  batch_size = tf.shape(output)[0]
  max_length = tf.shape(output)[1]
  out_size = int(output.get_shape()[2])
  index = tf.range(0, batch_size) * max_length + (seq_length - 1)
  flat = tf.reshape(output, [-1, out_size])
  relevant = tf.gather(flat, index)
  return relevant

#def mask_cost():
#sess = tf.Session()
#t = tf.constant([[[1,2],[3,4],[5,5]],[[2,2],[8,8],[9,9]]])
#l = tf.constant([1,2])
#o = last_relevant(t,l)
#print(sess.run(o))
