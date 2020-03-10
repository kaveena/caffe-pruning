#!/usr/bin/env python3
import caffe
import os
import struct
import sys
import random
import numpy as np
import argparse
import time

sys.dont_write_bytecode = True

def test(solver, itr, accuracy_layer_name, loss_layer_name):
  accuracy = dict()
  for i in range(itr):
    output = solver.test_nets[0].forward()
    for j in output.keys():
      if j in accuracy.keys():
        accuracy[j] = accuracy[j] + output[j]
      else:
        accuracy[j] = output[j].copy()

  for j in accuracy.keys():
    accuracy[j] /= float(itr)

  return accuracy[accuracy_layer_name]*100.0, accuracy[loss_layer_name]

def prune_weight(net, pruned_layer_name, pruned_weight_idx):
  layer = net.layer_dict[pruned_layer_name]
  p = pruned_weight_idx
  layer.blobs[0].data.flat[p] = 0

def prune_mask(net, pruned_layer_name, pruned_weight_idx):
  layer = net.layer_dict[pruned_layer_name]
  p = pruned_weight_idx
  layer.blobs[2].data.flat[p] = 0

def parser():
    parser = argparse.ArgumentParser(description='Caffe Weight Pruning Tool')
    parser.add_argument('--solver', action='store', default=None,
            help='the caffe solver to use')
    parser.add_argument('--model', action='store', default=None,
            help='model prototxt to use')
    parser.add_argument('--input', action='store', default=None,
            help='pretrained caffemodel')
    parser.add_argument('--output', action='store', default=None,
            help='output pruned caffemodel')
    parser.add_argument('--gpu', action='store_true', default=False,
            help='Use GPU')
    parser.add_argument('--verbose', action='store_true', default=False,
            help='Print summary of pruning process')
    parser.add_argument('--accuracy-layer-name', action='store', default='top-1',
            help='Name of layer computing accuracy')
    parser.add_argument('--loss-layer-name', action='store', default='loss',
            help='Name of layer computing loss')
    return parser

if __name__=='__main__':
  args = parser().parse_args()

  if args.solver is None:
    print("Caffe solver needed")
    exit(1)

  if args.output is None:
    print("Missing output caffemodel path")
    exit(1)

  if args.gpu:
    caffe.set_mode_gpu()
  else:
    caffe.set_mode_cpu()

  test_batches = 4

  net = caffe.Net(args.model, caffe.TEST)

  pruning_solver = caffe.SGDSolver(args.solver)
  pruning_solver.net.copy_from(args.input)
  pruning_solver.test_nets[0].share_with(pruning_solver.net)
  net.share_with(pruning_solver.net)

  layer_list = []
  layer_list += list(filter(lambda x: 'Convolution' in net.layer_dict[x].type, net.layer_dict.keys()))
  net_layers = net.layer_dict

  # We will have to keep re-checking this, so memoize it
  layer_weight_dims = dict()
  for layer in layer_list:
    l = net.layer_dict[layer]
    layer_weight_dims[layer] = l.blobs[0].shape

  # The pruning state is a list of the already-pruned weight positions for each layer
  prune_state = dict()
  for layer in layer_list:
    mask_data = net.layer_dict[layer].blobs[2].data
    prune_state[layer] = np.setdiff1d(np.nonzero(mask_data), np.arange(mask_data.size))

  # Get initial test accuraccy
  test_acc, ce_loss = test(pruning_solver, test_batches, args.accuracy_layer_name, args.loss_layer_name)

  if args.verbose:
    print("Initial test accuracy:", test_acc)
    sys.stdout.flush()

  while (test_acc > 40.0):
    # Generate a random subset of remaining weights to prune
    # Use one randomized pruning signal per layer
    pruning_signals = dict()

    for layer_name in layer_list:
      pruning_signals[layer_name] = np.zeros_like(net.layer_dict[layer_name].blobs[0].data)
      valid_indices = np.setdiff1d(np.arange(np.prod(layer_weight_dims[layer_name])), prune_state[layer_name])
      num_pruned_weights = np.random.randint(0, valid_indices.size*prune_factor)
      pruning_signals[layer_name] = np.random.choice(valid_indices, num_pruned_weights, replace=False)
      prune_state[layer_name] = np.union1d(prune_state[layer_name], pruning_signals[layer_name])

    # Now the actual pruning step
    for layer in layer_list:
      for weight_idx in pruning_signals[layer_name]:
        prune_mask(net, layer_name, weight_idx)

    test_acc, ce_loss = test(pruning_solver, test_batches, args.accuracy_layer_name, args.loss_layer_name)

    removed_weights = 0
    total_weights = 0
    for layer_name in layer_list:
      removed_weights += prune_state[layer_name].size
      total_weights += net.layer_dict[layer_name].blobs[0].data.size

      print("Test accuracy:", test_acc)
      print("Removed", removed_weights, "of", total_weights, "weights")
      sys.stdout.flush()

  pruning_solver.net.save(args.output)

  exit(0)
