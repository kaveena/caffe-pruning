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

def prune_weight(net, pruned_layer_idx, pruned_weight_idx):
  layer = net.layer_dict[pruned_layer_idx]
  m = layer.channels
  h = layer.height
  w = layer.width
  c = layer.blobs[0].data.shape[1]
  k = layer.blobs[0].data.shape[2]
  p = pruned_weight_idx
  layer.blobs[0].data[p/m][p/c][p/k][p/k] = 0

def prune_mask(net, pruned_layer_idx, pruned_weight_idx):
  layer = net.layer_dict[pruned_layer_idx]
  m = layer.channels
  h = layer.height
  w = layer.width
  c = layer.blobs[0].data.shape[1]
  k = layer.blobs[0].data.shape[2]
  p = pruned_weight_idx
  layer.blobs[layer.mask_pos_].data[p/m][p/c][p/k][p/k] = 0

def parser():
    parser = argparse.ArgumentParser(description='Caffe Output Channel Pruning Script')
    parser.add_argument('--solver', action='store', default=None,
            help='the caffe solver to use')
    parser.add_argument('--model', action='store', default=None,
            help='model prototxt to use')
    parser.add_argument('--input', action='store', default=None,
            help='pretrained caffemodel')
    parser.add_argument('--output', action='store', default=None,
            help='output pruned caffemodel')
    parser.add_argument('--finetune', action='store_true', default=False,
            help='finetune the pruned network')
    parser.add_argument('--stop-accuracy', type=float, default='10.0',
            help='Stop pruning when test accuracy drops below this value')
    parser.add_argument('--test-batches', type=int, default=80,
            help='Number of batches to use for testing')
    parser.add_argument('--finetune-batches', type=int, default=200,
            help='Number of batches to use for finetuning')
    parser.add_argument('--test-interval', type=int, default=1,
            help='After how many pruning steps to test')
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

  if args.output_weights is None:
    print("Missing output caffemodel path")
    exit(1)

  if args.gpu:
    caffe.set_mode_gpu()
  else:
    caffe.set_mode_cpu()

  net = caffe.Net(args.model, caffe.TEST)

  pruning_solver = caffe.SGDSolver(args.solver)
  pruning_solver.net.copy_from(args.pretrained)
  pruning_solver.test_nets[0].share_with(pruning_solver.net)
  net.share_with(pruning_solver.net)

  convolution_list = list(filter(lambda x: 'Convolution' in net.layer_dict[x].type, net.layer_dict.keys()))
  net_layers = net.layer_dict

  # The pruning state is a list of the already-pruned weight positions for each layer
  prune_state = dict()
  for layer in convolution_list:
    prune_state[layer] = []

  # We will have to keep re-checking this, so memoize it
  layer_weight_dims = dict()
  for layer in convolution_list:
    layer_weight_dims[layer] = (layer.channels, layer.blobs[0].data.shape[1], layer.blobs[0].data.shape[2], layer.blobs[0].data.shape[3])

  # Get initial test accuraccy
  test_acc, ce_loss = test(pruning_solver, args.test_size, args.acc_layer, args.loss_layer)

  while (test_acc >= args.stop_accuracy):
    pruning_solver.net.save(args.output_weights)

    # Generate a random subset of remaining weights to prune
    # Use one randomized pruning signal per layer
    pruning_signals = dict()
    for layer in convolution_list:
      pruning_signals[layer] = np.zeros_like(layer.blobs[0].data)
      d = layer_weight_dims[layer]
      valid_indices = np.setdiff1d(np.arange(d[0]*d[1]*d[2]*d[3]), prune_state[layer])
      num_pruned_weights = np.random.randint(0, valid_indices.size)
      pruning_signals[layer] = np.random.choice(num_pruned_weights, valid_indices, replace=False)
      prune_state[layer] = np.union1d(prune_state[layer], pruning_signals[layer])

      if args.verbose:
        print(layer, pruning_signals[layer])

    # Now the actual pruning step
    for layer in convolution_list:
      for weight_idx in pruning_signals[layer]:
        prune_mask(net, layer, weight_idx)

    if args.finetune:
      pruning_solver.step(args.finetune_batches)

    test_acc, ce_loss = test(pruning_solver, args.test_batches, args.acc_layer, args.loss_layer)

  exit(0)
