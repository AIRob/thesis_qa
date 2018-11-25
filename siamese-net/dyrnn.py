import tensorflow as tf
# sequence : [batch_size*max_seq_len] with zero paddings
# return : [batch_size] : value of seq_length (not contains zero paddings)
def seq_length(sequence):
    used = tf.sign(tf.abs(sequence))
    length = tf.reduce_sum(used, 1)
    length = tf.cast(length, tf.int32)
    return length


#https://stackoverflow.com/questions/39111373/tensorflow-chaining-tf-gather-produces-indexedslices-warning
def last_relevant_gather(output, seq_length):
    batch_size = tf.shape(output)[0]
    max_length = tf.shape(output)[1]
    out_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (seq_length - 1)
    flat = tf.reshape(output, [-1, out_size])
    relevant = tf.gather(flat, index)
    return relevant


def last_relevant(output, seq_length):
    batch_size = tf.shape(output)[0]
    max_length = tf.shape(output)[1]
    out_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (seq_length - 1)
    flat = tf.reshape(output, [-1, out_size])
    zero_one = tf.one_hot(index,tf.size(flat)//int(flat.get_shape()[1]))
    zero_one = tf.cast(zero_one,tf.int32)
    zero_one = tf.reduce_sum(zero_one, 0)
    relevant  = tf.dynamic_partition(flat,zero_one,2)  
    return relevant[1]

#def mask_cost():
#pass
#sess = tf.Session()
#t = tf.constant([[[1,2],[3,4],[5,5]],[[2,2],[8,8],[9,9]]])
#l = tf.constant([1,2])
#o1 = last_relevant(t,l)
#o2 =   last_relevant_gather(t,l)
#print(sess.run(o1))
#print(sess.run(o2))
#sess = tf.Session()
#t = tf.constant([1,2,3])
#l = tf.constant([1])
#zero_one = tf.one_hot(l,3)
#zero_one = tf.Print(zero_one,[zero_one],message="123")
#zero_one = tf.cast(zero_one,tf.int32)
#zero_one = tf.Print(zero_one,[zero_one],message="123")
#zero_one = tf.reduce_sum(zero_one, 0)
#zero_one = tf.Print(zero_one,[zero_one],message="123")
#o = tf.dynamic_partition(t,zero_one,2)
#print(sess.run(o))