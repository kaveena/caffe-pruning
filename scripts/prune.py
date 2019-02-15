import caffe
import os
import struct
import sys
import random
import numpy as np
import argparse
import time

sys.dont_write_bytecode = True

_caffe_saliencies_ = caffe._caffe.SALIENCY.names
_caffe_saliency_input_ = caffe._caffe.SALIENCY_INPUT.names
_caffe_saliency_norm_ = caffe._caffe.SALIENCY_NORM.names

l0_normalisation = lambda x, n, m, h, w, c, k, input_type : x/float(c * k * k) if (input_type=="WEIGHT") else x/float(h * w)
l1_normalisation = lambda x, n, m, h, w, c, k, input_type : x/float(np.abs(x).sum()) if float(np.abs(x).sum()) != 0 else x
l2_normalisation = lambda x, n, m, h, w, c, k, input_type : x/np.sqrt(np.power(x,2).sum()) if float(np.sqrt(np.power(x,2).sum())) != 0 else x
no_normalisation = lambda x, n, m, h, w, c, k, input_type : x

def parser():
    parser = argparse.ArgumentParser(description='Caffe Output Channel Pruning Script')
    parser.add_argument('--solver', action='store', default=None,
            help='the caffe solver to use')
    parser.add_argument('--model', action='store', default=None,
            help='model prototxt to use')
    parser.add_argument('--weights', action='store', default=None,
            help='pretrained caffemodel')
    parser.add_argument('--output-weights', action='store', default=None,
            help='output pruned caffemodel')
    parser.add_argument('--retrain', action='store_true', default=False,
            help='retrain the pruned network')
    parser.add_argument('--prune', action='store_true', default=False,
            help='prune network')
    parser.add_argument('--filename', action='store', default='summary_',
            help='prefix for storing pruning data')
    parser.add_argument('--stop-accuracy', type=float, default='10.0',
            help='Stop pruning when test accuracy drops below this value')
    parser.add_argument('--method', action='store', default='FISHER',
            help='Pruning method')
    parser.add_argument('--saliency-norm', action='store', default='NONE',
            help='Caffe saliency_norm')
    parser.add_argument('--saliency-input', action='store', default='ACTIVATION',
            help='Caffe saliency_input')
    parser.add_argument('--normalisation', action='store', default='no_normalisation',
            help='Layer-wise normalisation to use for saliency')
    parser.add_argument('--test-size', type=int, default=80,
            help='Number of batches to use for testing')
    parser.add_argument('--train-size', type=int, default=200,
            help='Number of batches to use for training')
    parser.add_argument('--eval-size', type=int, default=40,
            help='Number of batches to use for evaluating the saliency')
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

def update_mask(net, pruned_channel, convolution_list, channels, prune=True, final=True):
  fill = 0 if prune else 1
  idx = np.where(channels>pruned_channel)[0][0]
  idx_convolution = convolution_list[idx]
  idx_channel = (pruned_channel - channels[idx-1]) if idx > 0 else pruned_channel
  conv_module = net.layer_dict[idx_convolution]
  bias = conv_module.bias_term_
  if final and prune:
    conv_module.blobs[0].data[idx_channel].fill(0)
  if final and prune and bias:
    conv_module.blobs[1].data[idx_channel] = 0
  conv_module.blobs[conv_module.mask_pos_].data[idx_channel].fill(fill)
  if bias:
    conv_module.blobs[conv_module.mask_pos_+1].data[idx_channel] = 0

if __name__=='__main__':
  start = time.time()

  args = parser().parse_args()

  if args.solver is None:
    print("Caffe solver needed")
    exit(1)

  if args.output_weights is None:
    print("Missing output caffemodel path")
    exit(1)

  if args.gpu:
    caffe.set_mode_gpu()

  net = caffe.Net(args.model, caffe.TEST)

  saliency_solver = caffe.SGDSolver(args.solver)
  saliency_solver.net.copy_from(args.pretrained)
  saliency_solver.test_nets[0].share_with(saliency_solver.net)
  net.share_with(saliency_solver.net)

  convolution_list = list(filter(lambda x: 'Convolution' in net.layer_dict[x].type, net.layer_dict.keys()))
  named_modules = net.layer_dict

  if args.method in _caffe_saliencies_.keys():
    for layer in convolution_list:
      conv_module = named_modules[layer]
      conv_module.saliency_ = _caffe_saliencies_[args.method]
      conv_module.saliency_norm_ = _caffe_saliency_norm_[args.saliency_norm]
      conv_module.saliency_input_ = _caffe_saliency_input_[args.saliency_input]

  total_channels = 0
  channels = []

  for layer in convolution_list:
    total_channels += named_modules[layer].blobs[0].num
    channels.append(named_modules[layer].blobs[0].num)
  channels = np.array(channels)
  channels = np.cumsum(channels)

  if args.verbose:
    print('Total number of channels to be considered for pruning: ', total_channels)

  active_channel = list(range(total_channels))
  test_acc, ce_loss = test(saliency_solver, args.test_size)

  # Check if net is already broken
  test_acc, ce_loss = test(saliency_solver, args.test_size, args.acc_layer, args.loss_layer)

  while (test_acc >= args.stop_accuracy):
    saliency_solver.net.save(args.output_weights)

    for j in range(total_channels):
    if args.method in _caffe_saliencies_:
      for layer in convolution_list:
        named_modules[layer].blobs[named_modules[layer].saliency_pos_].data.fill(0) # reset saliency

    pruning_signal = np.array([])

    # compute initial saliencies
    evalset_size = args.eval_size;
    for iter in range(evalset_size):
      if args.method == 'random':
        break
      net.forward()
      net.backward()
      if (args.method == 'WEIGHT_AVG') and (args.saliency_input == 'WEIGHT'):
        break   #no need to do multiple passes of the network

      if (args.method == 'apoz'):
        pruning_signal_partial = np.array([])
        if (args.saliency_input == 'ACTIVATION'):
          for layer in convolution_list:
            caffe_layer = net.blobs[layer]
            n = caffe_layer.num
            m = caffe_layer.channels
            h = caffe_layer.height
            w = caffe_layer.width
            c = named_modules[layer].blobs[0].data.shape[1]
            k = named_modules[layer].blobs[0].data.shape[2]
            saliency_data = (net.blobs[layer].data > 0.0).sum(axis=(0,2,3)) / float( n * h * w)

            if args.normalization == 'l0' :
              saliency_normalised = l0_normalisation(saliency_data, n, m, h , w, c , k, args.saliency_input)
            if args.normalization == 'l1' :
              saliency_normalised = l1_normalisation(saliency_data, n, m, h , w, c , k, args.saliency_input)
            if args.normalization == 'l2' :
              saliency_normalised = l2_normalisation(saliency_data, n, m, h , w, c , k, args.saliency_input)
            if args.normalization == 'no' :
              saliency_normalised = no_normalisation(saliency_data, n, m, h , w, c , k, args.saliency_input)

            pruning_signal_partial = np.hstack([pruning_signal_partial, saliency_normalised])
          if iter == 0:
            pruning_signal = pruning_signal_partial.data
          else:
            pruning_signal += pruning_signal_partial

        else:
          print('Not implemented')
          exit(1)

    if (args.method != 'WEIGHT_AVG') or (args.saliency_input != 'WEIGHT'):
      pruning_signal /= float(evalset_size) # get approximate change in loss using taylor expansions

    if args.method in _caffe_saliencies_:
      for layer in convolution_list:
        saliency_data = named_modules[layer].blobs[named_modules[layer].saliency_pos_].data[0]
        caffe_layer = net.blobs[layer]
        n = caffe_layer.num
        m = caffe_layer.channels
        h = caffe_layer.height
        w = caffe_layer.width
        c = named_modules[layer].blobs[0].data.shape[1]
        k = named_modules[layer].blobs[0].data.shape[2]

        if args.normalization == 'l0' :
          saliency_normalised = l0_normalisation(saliency_data, n, m, h , w, c , k, args.saliency_input)
        if args.normalization == 'l1' :
          saliency_normalised = l1_normalisation(saliency_data, n, m, h , w, c , k, args.saliency_input)
        if args.normalization == 'l2' :
          saliency_normalised = l2_normalisation(saliency_data, n, m, h , w, c , k, args.saliency_input)
        if args.normalization == 'no' :
          saliency_normalised = no_normalisation(saliency_data, n, m, h , w, c , k, args.saliency_input)

        pruning_signal = np.hstack([pruning_signal, saliency_normalised])

    if args.method == 'random':
      pruning_signal = np.zeros(total_channels)
      pruning_signal[random.sample(active_channel, 1)] = -1

    prune_channel_idx = np.argmin(pruning_signal[active_channel])
    prune_channel = active_channel[prune_channel_idx]
    update_mask(net, prune_channel, convolution_list, channels, final=True)

    if args.retrain:
      saliency_solver.step(args.train_size)

    if args.verbose:
      print(args.normalisation, args.method, 'Removed Channel: ', prune_channel, '  ||  Test Accuracy: ', test_acc)

    active_channel.remove(prune_channel)

  exit(0)
