from __future__ import division
from __future__ import print_function
from operator import itemgetter
from itertools import combinations
import time
import os

import tensorflow as tf
import numpy as np
import networkx as nx
import scipy.sparse as sp
import scipy.io as sio
from sklearn import metrics

import matplotlib.pyplot as plt
import pandas as pd
import h5py
# from skchem.metrics import bedroc_score

# newly added --------------
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from absl import app
from absl import flags
from absl import logging
FLAGS = flags.FLAGS
# ----------------------------

import pickle
from decagon.deep.optimizer import DecagonOptimizer
from decagon.deep.model import DecagonModel
from decagon.deep.minibatch import EdgeMinibatchIterator
from decagon.utility import rank_metrics, preprocessing

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

config = tf.compat.v1.ConfigProto()
# config = tf.ConfigProto()
config.gpu_options.allow_growth = True

np.random.seed(0)

def tsne_visualization(matrix):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    plt.figure(dpi=300)
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, random_state=0,
            n_iter=1000)
    tsne_results = tsne.fit_transform(matrix)
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def draw_graph(adj_matrix):
    G = nx.from_scipy_sparse_matrix(adj_matrix)
    pos = nx.spring_layout(G, iterations=100)
    d = dict(nx.degree(G))
    nx.draw(G, pos, node_color=range(3215), nodelist=d.keys(), 
        node_size=[v*20+20 for v in d.values()], cmap=plt.cm.Dark2)
    plt.show()

def get_accuracy_scores(feed_dict, edges_pos, edges_neg, edge_type, placeholders, minibatch, sess, opt, name=None, return_preds=False):
    feed_dict.update({placeholders['dropout']: 0})
    feed_dict.update({placeholders['batch_edge_type_idx']: minibatch.edge_type2idx[edge_type]})
    feed_dict.update({placeholders['batch_row_edge_type']: edge_type[0]})
    feed_dict.update({placeholders['batch_col_edge_type']: edge_type[1]})
    rec = sess.run(opt.predictions, feed_dict=feed_dict)

    def sigmoid(x):
        return 1. / (1 + np.exp(-x))

    preds = []
    actual = []
    predicted = []
    edge_ind = 0
    for u, v in edges_pos[edge_type[:2]][edge_type[2]]:
        score = sigmoid(rec[u, v])
        preds.append(score)

        assert adj_mats_orig[edge_type[:2]][edge_type[2]][u,v] == 1, 'Problem 1'

        actual.append(edge_ind)
        predicted.append((score, edge_ind))
        edge_ind += 1

    preds_neg = []
    for u, v in edges_neg[edge_type[:2]][edge_type[2]]:
        score = sigmoid(rec[u, v])
        preds_neg.append(score)
        assert adj_mats_orig[edge_type[:2]][edge_type[2]][u,v] == 0, 'Problem 0'

        predicted.append((score, edge_ind))
        edge_ind += 1

    preds_all = np.hstack([preds, preds_neg])
    preds_all = np.nan_to_num(preds_all)
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    predicted = list(zip(*sorted(predicted, reverse=True, key=itemgetter(0))))[1]

    roc_sc = metrics.roc_auc_score(labels_all, preds_all)
    aupr_sc = metrics.average_precision_score(labels_all, preds_all)
    apk_sc = rank_metrics.apk(actual, predicted, k=200)
    # bedroc_sc = bedroc_score(labels_all, preds_all)
    if name!=None:
        with open(name, 'wb') as f:
            pickle.dump([labels_all, preds_all], f)
    # ********* I add it to try plotting graphs ***************
    if return_preds:
        labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
        preds_all = np.hstack([preds, preds_neg])
        return roc_sc, aupr_sc, apk_sc, labels_all, preds_all
    # return roc_sc, aupr_sc, apk_sc, bedroc_sc
    return roc_sc, aupr_sc, apk_sc

# ********* I add it to try plotting graphs ***************
def plot_roc_pr_curves(labels, preds):
    fpr, tpr, _ = metrics.roc_curve(labels, preds)
    roc_auc = metrics.auc(fpr, tpr)

    precision, recall, _ = metrics.precision_recall_curve(labels, preds)
    pr_auc = metrics.average_precision_score(labels, preds)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    plt.figure()
    plt.plot(recall, precision, color='blue', lw=2, label='PR curve (area = %0.2f)' % pr_auc)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.show()
# ---------------------------------------------------------------

def construct_placeholders(edge_types):
    placeholders = {
        'batch': tf.placeholder(tf.int32, name='batch'),
        'batch_edge_type_idx': tf.placeholder(tf.int32, shape=(), name='batch_edge_type_idx'),
        'batch_row_edge_type': tf.placeholder(tf.int32, shape=(), name='batch_row_edge_type'),
        'batch_col_edge_type': tf.placeholder(tf.int32, shape=(), name='batch_col_edge_type'),
        'degrees': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
    }
    placeholders.update({
        'adj_mats_%d,%d,%d' % (i, j, k): tf.sparse_placeholder(tf.float32)
        for i, j in edge_types for k in range(edge_types[i,j])})
    placeholders.update({
        'feat_%d' % i: tf.sparse_placeholder(tf.float32)
        for i, _ in edge_types})
    return placeholders

def network_edge_threshold(network_adj, threshold):
    edge_tmp, edge_value, shape_tmp = preprocessing.sparse_to_tuple(network_adj)
    preserved_edge_index = np.where(edge_value>threshold)[0]
    preserved_network = sp.csr_matrix(
        (edge_value[preserved_edge_index], 
        (edge_tmp[preserved_edge_index,0], edge_tmp[preserved_edge_index, 1])),
        shape=shape_tmp)
    return preserved_network


def get_prediction(feed_dict, placeholders, minibatch, sess, opt, edges_pos, edges_neg, edge_type):
    feed_dict.update({placeholders['dropout']: 0})
    feed_dict.update({placeholders['batch_edge_type_idx']: minibatch.edge_type2idx[edge_type]})
    feed_dict.update({placeholders['batch_row_edge_type']: edge_type[0]})
    feed_dict.update({placeholders['batch_col_edge_type']: edge_type[1]})
    
    
    tiff_check = sess.run(opt.cost, feed_dict=feed_dict)
    print("!!!~~~!!!", tiff_check)
    
    rec = sess.run(opt.predictions, feed_dict=feed_dict)

    return 1. / (1 + np.exp(-rec))

gene_phenes_path = './data_prioritization/genes_phenes.mat'
f = h5py.File(gene_phenes_path, 'r')
gene_network_adj = sp.csc_matrix((np.array(f['GeneGene_Hs']['data']),
    np.array(f['GeneGene_Hs']['ir']), np.array(f['GeneGene_Hs']['jc'])),
    shape=(12331,12331))
gene_network_adj = gene_network_adj.tocsr()
disease_network_adj = sp.csc_matrix((np.array(f['PhenotypeSimilarities']['data']),
    np.array(f['PhenotypeSimilarities']['ir']), np.array(f['PhenotypeSimilarities']['jc'])),
    shape=(3215, 3215))
disease_network_adj = disease_network_adj.tocsr()

disease_network_adj = network_edge_threshold(disease_network_adj, 0.2)


dg_ref = f['GenePhene'][0][0]
gene_disease_adj = sp.csc_matrix((np.array(f[dg_ref]['data']),
    np.array(f[dg_ref]['ir']), np.array(f[dg_ref]['jc'])),
    shape=(12331, 3215))
gene_disease_adj = gene_disease_adj.tocsr()

novel_associations_adj = sp.csc_matrix((np.array(f['NovelAssociations']['data']),
    np.array(f['NovelAssociations']['ir']), np.array(f['NovelAssociations']['jc'])),
    shape=(12331,3215))

gene_feature_path = './data_prioritization/GeneFeatures.mat'
f_gene_feature = h5py.File(gene_feature_path,'r')
gene_feature_exp = np.array(f_gene_feature['GeneFeatures'])
gene_feature_exp = np.transpose(gene_feature_exp)
gene_network_exp = sp.csc_matrix(gene_feature_exp)

row_list = [3215, 1137, 744, 2503, 1143, 324, 1188, 4662, 1243]
gene_feature_list_other_spe = list()
for i in range(1,9):
    dg_ref = f['GenePhene'][i][0]
    disease_gene_adj_tmp = sp.csc_matrix((np.array(f[dg_ref]['data']),
        np.array(f[dg_ref]['ir']), np.array(f[dg_ref]['jc'])),
        shape=(12331, row_list[i]))
    gene_feature_list_other_spe.append(disease_gene_adj_tmp)

disease_tfidf_path = './data_prioritization/clinicalfeatures_tfidf.mat'
f_disease_tfidf = h5py.File(disease_tfidf_path)
disease_tfidf = np.array(f_disease_tfidf['F'])
disease_tfidf = np.transpose(disease_tfidf)
disease_tfidf = sp.csc_matrix(disease_tfidf)

dis_dis_adj_list= list()
dis_dis_adj_list.append(disease_network_adj)

val_test_size = 0.1
n_genes = 12331
n_dis = 3215
n_dis_rel_types = len(dis_dis_adj_list)
gene_adj = gene_network_adj
gene_degrees = np.array(gene_adj.sum(axis=0)).squeeze()

gene_dis_adj = gene_disease_adj
dis_gene_adj = gene_dis_adj.transpose(copy=True)

dis_degrees_list = [np.array(dis_adj.sum(axis=0)).squeeze() for dis_adj in dis_dis_adj_list]

adj_mats_orig = {
    (0, 0): [gene_adj, gene_adj.transpose(copy=True)],
    (0, 1): [gene_dis_adj],
    (1, 0): [dis_gene_adj],
    (1, 1): dis_dis_adj_list + [x.transpose(copy=True) for x in dis_dis_adj_list],
}
degrees = {
    0: [gene_degrees, gene_degrees],
    1: dis_degrees_list + dis_degrees_list,
}

gene_feat = sp.hstack(gene_feature_list_other_spe+[gene_feature_exp])
gene_nonzero_feat, gene_num_feat = gene_feat.shape
gene_feat = preprocessing.sparse_to_tuple(gene_feat.tocoo())

dis_feat = disease_tfidf
dis_nonzero_feat, dis_num_feat = dis_feat.shape
dis_feat = preprocessing.sparse_to_tuple(dis_feat.tocoo())

num_feat = {
    0: gene_num_feat,
    1: dis_num_feat,
}
nonzero_feat = {
    0: gene_nonzero_feat,
    1: dis_nonzero_feat,
}
feat = {
    0: gene_feat,
    1: dis_feat,
}

edge_type2dim = {k: [adj.shape for adj in adjs] for k, adjs in adj_mats_orig.items()}
# edge_type2decoder = {
#     (0, 0): 'bilinear',
#     (0, 1): 'bilinear',
#     (1, 0): 'bilinear',
#     (1, 1): 'bilinear',
# }

edge_type2decoder = {
    (0, 0): 'innerproduct',
    (0, 1): 'innerproduct',
    (1, 0): 'innerproduct',
    (1, 1): 'innerproduct',
}

edge_types = {k: len(v) for k, v in adj_mats_orig.items()}
num_edge_types = sum(edge_types.values())
print("Edge types:", "%d" % num_edge_types)

# ------------- I made some changes ---------------------
flags.DEFINE_integer('neg_sample_size', 1, 'Negative sample size.')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('hidden1', 64, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_float('weight_decay', 0.001, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0.1, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('max_margin', 0.1, 'Max margin parameter in hinge loss')
flags.DEFINE_integer('batch_size', 512, 'minibatch size.')
flags.DEFINE_boolean('bias', True, 'Bias term.')

def main(argv):
    try:
        FLAGS = flags.FLAGS
        FLAGS(argv)
        logging.info('Negative sample size: %d', FLAGS.neg_sample_size)
        logging.info('Learning rate: %f', FLAGS.learning_rate)
        logging.info('Number of units in hidden layer 1: %d', FLAGS.hidden1)
        logging.info('Number of units in hidden layer 2: %d', FLAGS.hidden2)
        logging.info('Weight for L2 loss on embedding matrix: %f', FLAGS.weight_decay)
        logging.info('Dropout rate (1 - keep probability): %f', FLAGS.dropout)
        logging.info('Max margin parameter in hinge loss: %f', FLAGS.max_margin)
        logging.info('Minibatch size: %d', FLAGS.batch_size)
        logging.info('Bias term: %s', FLAGS.bias)

        print("Defining placeholders")
        placeholders = construct_placeholders(edge_types)

        print("Create minibatch iterator")
        minibatch = EdgeMinibatchIterator(
            adj_mats=adj_mats_orig,
            feat=feat,
            edge_types=edge_types,
            batch_size=FLAGS.batch_size,
            val_test_size=val_test_size
        )

        print("Create model")
        model = DecagonModel(
            placeholders=placeholders,
            num_feat=num_feat,
            nonzero_feat=nonzero_feat,
            edge_types=edge_types,
            decoders=edge_type2decoder,
        )

        print("Create optimizer")
        with tf.name_scope('optimizer'):
            opt = DecagonOptimizer(
                embeddings=model.embeddings,
                latent_inters=model.latent_inters,
                latent_varies=model.latent_varies,
                degrees=degrees,
                edge_types=edge_types,
                edge_type2dim=edge_type2dim,
                placeholders=placeholders,
                batch_size=FLAGS.batch_size,
                margin=FLAGS.max_margin
            )

        print("Initialize session")
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        feed_dict = {}
        saver = tf.train.Saver()
        saver.restore(sess,'./model/model.ckpt')
        feed_dict = minibatch.next_minibatch_feed_dict(placeholders=placeholders)
        feed_dict = minibatch.update_feed_dict(
            feed_dict=feed_dict,
            dropout=FLAGS.dropout,
            placeholders=placeholders)

        # temporarily removed bedroc
        roc_score, auprc_score, apk_score, labels_all, preds_all = get_accuracy_scores(feed_dict,
            minibatch.test_edges, minibatch.test_edges_false, minibatch.idx2edge_type[3], 
            placeholders, minibatch, sess, opt, return_preds=True)
        # roc_score, auprc_score, apk_score, bedroc = get_accuracy_scores(feed_dict,
        #     minibatch.test_edges, minibatch.test_edges_false, minibatch.idx2edge_type[3], 
        #     placeholders, minibatch, sess, opt)
    
        plot_roc_pr_curves(labels_all, preds_all)

        print("Edge type=", "[%02d, %02d, %02d]" % minibatch.idx2edge_type[3])
        print("Edge type:", "%04d" % 3, "Test AUROC score", "{:.5f}".format(roc_score))
        print("Edge type:", "%04d" % 3, "Test AUPRC score", "{:.5f}".format(auprc_score))
        print("Edge type:", "%04d" % 3, "Test AP@k score", "{:.5f}".format(apk_score))
        # print("Edge type:", "%04d" % 3, "Test BEDROC score", "{:.5f}".format(bedroc))
        print()

        prediction = get_prediction(feed_dict, placeholders, minibatch, sess, opt, 
        minibatch.test_edges, minibatch.test_edges_false, minibatch.idx2edge_type[3])

        print('Saving result...')
        np.save('./result/prediction.npy', prediction)

    except tf.errors.InvalidArgumentError as e:
        print(".")
    except AttributeError as e:
        print(".")
if __name__ == '__main__':
    app.run(main)

    # flags = tf.app.flags
    # FLAGS = flags.FLAGS