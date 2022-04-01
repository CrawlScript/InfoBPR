# coding=utf-8
import tensorflow as tf
import numpy as np

def tf_info_bpr(a_embeddings, b_embeddings, pos_edges, num_negs=300):

    if isinstance(pos_edges, list):
        pos_edges = np.array(pos_edges)

    a_indices = pos_edges[:, 0]
    b_indices = pos_edges[:, 1]

    num_pos_edges = tf.shape(pos_edges)[0]


    num_b = tf.shape(b_embeddings)[0]
    neg_b_indices = tf.random.uniform([num_pos_edges, num_negs], 0, num_b, dtype=tf.int32)

    embedded_a = tf.gather(a_embeddings, a_indices)
    embedded_b = tf.gather(b_embeddings, b_indices)
    embedded_neg_b = tf.gather(b_embeddings, neg_b_indices)
    

    embedded_combined_b = tf.concat(
            [tf.expand_dims(embedded_b, axis=1), embedded_neg_b], 
            axis=1
            )

    logits = tf.squeeze(embedded_neg_b @ tf.expand_dims(embedded_a, axis=-1), axis=-1)

    info_bpr_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits,
            labels=tf.ones_like(a_indices, dtype=tf.int64)
            )

    return info_bpr_loss



    

    
