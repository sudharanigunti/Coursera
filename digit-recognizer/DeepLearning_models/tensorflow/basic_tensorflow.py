import tensorflow as tf

a = tf.constant([2], name = "constant_a")
b = tf.constant([3], name = "constant_b") 

print(a)                # gives whole defination ex: tf.Tensor([2], shape=(1,), dtype=int32)
tf.print(a.numpy()[0])  # gives values as output ex: 2


@tf.function            # TF autograph transforms below function into T.F control flow 
def add(a,b):
    c = tf.add(a,b)     # you can also use c = a + b 
    print(c)            # Output: Tensor("Add:0", shape=(1,), dtype=int32)
    return c

result = add(a,b)
tf.print(result[0])     # Output: 5
