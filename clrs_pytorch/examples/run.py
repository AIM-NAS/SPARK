# Copyright 2022 DeepMind Technologies Limited. All Rights Reserved.
#
# Apache-2.0 License
#
# 本文件基于原始 run.py，**仅在涉及 TensorFlow / Haiku / JAX 的位置做 PyTorch 化改写**，
# 其余逻辑、函数结构、flag 名称与原始脚本保持一一对应（尽量“行对行”）。
#
# 主要改动点：
# - 去掉 TensorFlow 依赖与 with tf.device('/cpu:0') 语句（PyTorch 下不需要）。
# - 去掉 JAX RNG；改用 numpy.RandomState + 递增 seed int 传递；
#   若你的 clrs_pytorch BaselineModel 需要 torch.Generator，可将该 int 转成对应 generator。
# - clrs → clrs_pytorch：processor_factory / build_sampler / evaluate / dataset 工具等。
# - jax.tree_util.tree_map + np.concatenate → 递归拼接工具 _concat_tree。
# - 保留 flags / 训练-评估流程 / 日志文本 / 采样器创建逻辑与参数对齐。

from __future__ import annotations
import torch
torch.backends.cudnn.benchmark = True                 # 根据卷积形状自动选最快算法
torch.backends.cuda.matmul.allow_tf32 = True          # A100/3090: TF32张量核
torch.set_float32_matmul_precision('high')            # matmul 走 TF32/高精度核
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler(enabled=True)   # 放在 main() 开头初始化一次

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import functools
import os
import shutil
from typing import Any, Dict, List, Optional

from absl import app
from absl import flags
from absl import logging
import inspect
import numpy as np
import requests
import torch

# === 关键替换：使用 clrs_pytorch 而非 clrs / haiku / jax ===
import clrs_pytorch as clrs

print("processor_factory from:", inspect.getsourcefile(clrs.get_processor_factory))
print("BaselineModel from:", inspect.getsourcefile(clrs.models.BaselineModel))
print("build_sampler from:", inspect.getsourcefile(clrs.build_sampler))
print("evaluate from:", inspect.getsourcefile(clrs.evaluate))

# -------------------- Flags（保持与原版一致） --------------------
flags.DEFINE_list('algorithms', ['dfs'], 'Which algorithms to run.')
flags.DEFINE_list('train_lengths', ['4', '7', '11', '13', '16'],
                  'Which training sizes to use. A size of -1 means '
                  'use the benchmark dataset.')
flags.DEFINE_integer('length_needle', -8,
                     'Length of needle for training and validation '
                     '(not testing) in string matching algorithms. '
                     'A negative value randomizes the length for each sample '
                     'between 1 and the opposite of the value. '
                     'A value of 0 means use always 1/4 of the length of '
                     'the haystack (the default sampler behavior).')
flags.DEFINE_integer('seed', 42, 'Random seed to set')

flags.DEFINE_boolean('random_pos', True,
                     'Randomize the pos input common to all algos.')
flags.DEFINE_boolean('enforce_permutations', True,
                     'Whether to enforce permutation-type node pointers.')
flags.DEFINE_boolean('enforce_pred_as_input', True,
                     'Whether to change pred_h hints into pred inputs.')
flags.DEFINE_integer('batch_size', 32, 'Batch size used for training.')
flags.DEFINE_boolean('chunked_training', False,
                     'Whether to use chunking for training.')
flags.DEFINE_integer('chunk_length', 16,
                     'Time chunk length used for training (if '
                     '`chunked_training` is True.')
flags.DEFINE_integer('train_steps', 10000, 'Number of training iterations.')
flags.DEFINE_integer('eval_every', 50, 'Evaluation frequency (in steps).')
flags.DEFINE_integer('test_every', 500, 'Evaluation frequency (in steps).')

flags.DEFINE_integer('hidden_size', 128,
                     'Number of hidden units of the model.')
flags.DEFINE_integer('nb_heads', 1, 'Number of heads for GAT processors')
flags.DEFINE_integer('nb_msg_passing_steps', 1,
                     'Number of message passing steps to run per hint.')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate to use.')
flags.DEFINE_float('grad_clip_max_norm', 1.0,
                   'Gradient clipping by norm. 0.0 disables grad clipping')
flags.DEFINE_float('dropout_prob', 0.0, 'Dropout rate to use.')
flags.DEFINE_float('hint_teacher_forcing', 0.0,
                   'Probability that ground-truth teacher hints are encoded '
                   'during training instead of predicted hints. Only '
                   'pertinent in encoded_decoded modes.')
flags.DEFINE_enum('hint_mode', 'encoded_decoded',
                  ['encoded_decoded', 'decoded_only', 'none'],
                  'How should hints be used? Note, each mode defines a '
                  'separate task, with various difficulties. `encoded_decoded` '
                  'requires the model to explicitly materialise hint sequences '
                  'and therefore is hardest, but also most aligned to the '
                  'underlying algorithmic rule. Hence, `encoded_decoded` '
                  'should be treated as the default mode for our benchmark. '
                  'In `decoded_only`, hints are only used for defining '
                  'reconstruction losses. Often, this will perform well, but '
                  'note that we currently do not make any efforts to '
                  'counterbalance the various hint losses. Hence, for certain '
                  'tasks, the best performance will now be achievable with no '
                  'hint usage at all (`none`).')
flags.DEFINE_enum('hint_repred_mode', 'soft', ['soft', 'hard', 'hard_on_eval'],
                  'How to process predicted hints when fed back as inputs.'
                  'In soft mode, we use softmaxes for categoricals, pointers '
                  'and mask_one, and sigmoids for masks. '
                  'In hard mode, we use argmax instead of softmax, and hard '
                  'thresholding of masks. '
                  'In hard_on_eval mode, soft mode is '
                  'used for training and hard mode is used for evaluation.')
flags.DEFINE_boolean('use_ln', True,
                     'Whether to use layer normalisation in the processor.')
flags.DEFINE_boolean('use_lstm', False,
                     'Whether to insert an LSTM after message passing.')
flags.DEFINE_integer('nb_triplet_fts', 8,
                     'How many triplet features to compute?')

flags.DEFINE_enum('encoder_init', 'xavier_on_scalars',
                  ['default', 'xavier_on_scalars'],
                  'Initialiser to use for the encoders.')
flags.DEFINE_enum('processor_type', 'triplet_gmpnn',
                  ['deepsets', 'mpnn', 'pgn', 'pgn_mask',
                   'triplet_mpnn', 'triplet_pgn', 'triplet_pgn_mask',
                   'gat', 'gatv2', 'gat_full', 'gatv2_full',
                   'gpgn', 'gpgn_mask', 'gmpnn',
                   'triplet_gpgn', 'triplet_gpgn_mask', 'triplet_gmpnn'],
                  'Processor type to use as the network P.')

flags.DEFINE_string('checkpoint_path', '/tmp/CLRS301',
                    'Path in which checkpoints are saved.')
flags.DEFINE_string('dataset_path', '/data1/lz/clrs/openevolve/dataset/CLRS30_v1.0.0.tar/CLRS30_v1.0.0/clrs_dataset',
                    'Path in which dataset is stored.')
flags.DEFINE_boolean('freeze_processor', False,
                     'Whether to freeze the processor of the model.')

FLAGS = flags.FLAGS


PRED_AS_INPUT_ALGOS = [
    'binary_search', 'minimum', 'find_maximum_subarray',
    'find_maximum_subarray_kadane', 'matrix_chain_order', 'lcs_length',
    'optimal_bst', 'activity_selector', 'task_scheduling',
    'naive_string_matcher', 'kmp_matcher', 'jarvis_march']


def unpack(v):
  try:
    return v.item()
  except (AttributeError, ValueError):
    return v


def _iterate_sampler(sampler, batch_size):
  while True:
    yield sampler.next(batch_size)


# === Dataset 下载（保持原逻辑，clrs_pytorch 提供 get_clrs_folder 等价接口） ===
def _maybe_download_dataset(dataset_path):
  dataset_folder = os.path.join(dataset_path, clrs.get_clrs_folder())
  if os.path.isdir(dataset_folder):
    logging.info('Dataset found at %s. Skipping download.', dataset_folder)
    return dataset_folder
  logging.info('Dataset not found in %s. Downloading...', dataset_folder)

  clrs_url = clrs.get_dataset_gcp_url()
  request = requests.get(clrs_url, allow_redirects=True)
  clrs_file = os.path.join(dataset_path, os.path.basename(clrs_url))
  os.makedirs(dataset_folder)
  open(clrs_file, 'wb').write(request.content)
  shutil.unpack_archive(clrs_file, extract_dir=dataset_folder)
  os.remove(clrs_file)
  return dataset_folder


# === 递归拼接（替代 jax.tree_util.tree_map + np.concatenate） ===
def _concat_tree(list_of_trees, axis: int):
  """对由 DataPoint / list / tuple / dict 组成的树状结构逐叶子拼接。
     - 叶子是 torch.Tensor → torch.cat
     - 叶子是 np.ndarray   → np.concatenate
  """
  if not list_of_trees:
    return None

  t0 = list_of_trees[0]

  # 结构节点：list / tuple
  if isinstance(t0, (list, tuple)):
    zipped = list(zip(*list_of_trees))
    out = [_concat_tree(list(elems), axis) for elems in zipped]
    return type(t0)(out) if isinstance(t0, tuple) else out

  # 结构节点：dict
  if isinstance(t0, dict):
    keys = t0.keys()
    return {k: _concat_tree([t[k] for t in list_of_trees], axis) for k in keys}

  # 叶子节点：DataPoint
  try:
    from clrs_pytorch._src.probing import DataPoint
  except Exception:
    DataPoint = None

  if DataPoint is not None and isinstance(t0, DataPoint):
    datas = [t.data for t in list_of_trees]
    # torch.Tensor 走 torch.cat；否则走 numpy
    if any(isinstance(d, torch.Tensor) for d in datas):
      # 统一到第一个张量的 device/dtype
      first = next(d for d in datas if isinstance(d, torch.Tensor))
      dev, dt = first.device, first.dtype
      datas = [d.to(dev, dt) if isinstance(d, torch.Tensor)
               else torch.as_tensor(d, device=dev, dtype=dt)
               for d in datas]
      cat = torch.cat(datas, dim=axis)
    else:
      cat = np.concatenate(datas, axis=axis)
    return DataPoint(t0.name, t0.location, t0.type_, cat)

  # 叶子节点：torch.Tensor
  if isinstance(t0, torch.Tensor):
    # 统一 device/dtype 到第一个 tensor
    dev, dt = t0.device, t0.dtype
    ts = [x.to(dev, dt) if isinstance(x, torch.Tensor)
          else torch.as_tensor(x, device=dev, dtype=dt)
          for x in list_of_trees]
    return torch.cat(ts, dim=axis)

  # 叶子节点：numpy / 其他可转 numpy 的对象
  return np.concatenate(list_of_trees, axis=axis)


def _concat(dps, axis):
  return _concat_tree(dps, axis)


# === Collect & Eval（移除 JAX PRNG，改为 int seed 递增） ===

def collect_and_eval(sampler, predict_fn, sample_count, rng_seed_int, extras):
  processed_samples = 0
  preds = []
  outputs = []
  seed = int(rng_seed_int)
  while processed_samples < sample_count:
    feedback = next(sampler)
    batch_size = feedback.outputs[0].data.shape[0]
    outputs.append(feedback.outputs)

    cur_preds, _ = predict_fn(seed, feedback.features)
    preds.append(cur_preds)
    processed_samples += batch_size
    seed = (seed + 1) % (2**31 - 1)
  outputs = _concat(outputs, axis=0)
  preds = _concat(preds, axis=0)
  out = clrs.evaluate(outputs, preds)
  if extras:
    out.update(extras)
  return {k: unpack(v) for k, v in out.items()}


# === Sampler 构建（clrs_pytorch 等价 API） ===

def make_sampler(length: int,
                 rng: Any,
                 algorithm: str,
                 split: str,
                 batch_size: int,
                 multiplier: int,
                 randomize_pos: bool,
                 enforce_pred_as_input: bool,
                 enforce_permutations: bool,
                 chunked: bool,
                 chunk_length: int,
                 sampler_kwargs: Dict[str, Any]):

  if length < 0:  # load from file
    dataset_folder = _maybe_download_dataset(FLAGS.dataset_path)

    sampler, num_samples, spec = clrs.create_dataset(
        folder=dataset_folder,
        algorithm=algorithm,
        batch_size=batch_size,  # 这里通常已经按 batch 产出
        split=split
    )

  else:
    num_samples = clrs.CLRS30[split]['num_samples'] * multiplier

    sampler, spec = clrs.build_sampler(
        algorithm,
        seed=rng.randint(2**32),
        num_samples=num_samples,
        length=length,
        **sampler_kwargs,
    )
    sampler = _iterate_sampler(sampler, batch_size)

  if randomize_pos:
    sampler = clrs.process_random_pos(sampler, rng)
  if enforce_pred_as_input and algorithm in PRED_AS_INPUT_ALGOS:
    spec, sampler = clrs.process_pred_as_input(spec, sampler)
  spec, sampler = clrs.process_permutations(spec, sampler, enforce_permutations)
  if chunked:
    sampler = clrs.chunkify(sampler, chunk_length)
  return sampler, num_samples, spec


def make_multi_sampler(sizes, rng, **kwargs):
  ss = []
  tot_samples = 0
  for length in sizes:
    sampler, num_samples, spec = make_sampler(length, rng, **kwargs)
    ss.append(sampler)
    tot_samples += num_samples

  def cycle_samplers():
    while True:
      for s in ss:
        yield next(s)
  return cycle_samplers(), tot_samples, spec


# === 主流程（去掉 tf.device 上下文） ===

def create_samplers(
    rng,
    train_lengths: List[int],
    *,
    algorithms: Optional[List[str]] = None,
    val_lengths: Optional[List[int]] = None,
    test_lengths: Optional[List[int]] = None,
    train_batch_size: int = 32,
    val_batch_size: int = 32,
    test_batch_size: int = 32,
):
  train_samplers = []
  val_samplers = []
  val_sample_counts = []
  test_samplers = []
  test_sample_counts = []
  spec_list = []

  algorithms = algorithms or FLAGS.algorithms
  for algo_idx, algorithm in enumerate(algorithms):
    current_algo_train_lengths = train_lengths

    # NOTE: 原 code 在此有 with tf.device('/cpu:0')；PyTorch 无需此上下文，数据管道在 CPU 上默认即可。

    if algorithm in ['naive_string_matcher', 'kmp_matcher']:
      max_length = max(current_algo_train_lengths)
      if max_length > 0:
        max_length = (max_length * 5) // 4
      current_algo_train_lengths = [max_length]
      if FLAGS.chunked_training:
        current_algo_train_lengths = current_algo_train_lengths * len(current_algo_train_lengths)

    logging.info('Creating samplers for algo %s', algorithm)

    p = tuple([0.1 + 0.1 * i for i in range(9)])
    if p and algorithm in ['articulation_points', 'bridges', 'mst_kruskal', 'bipartite_matching']:
      p = tuple(np.array(p) / 2)
    length_needle = FLAGS.length_needle
    sampler_kwargs = dict(p=p, length_needle=length_needle)
    if length_needle == 0:
      sampler_kwargs.pop('length_needle')

    common_sampler_args = dict(
        algorithm=algorithms[algo_idx],
        rng=rng,
        enforce_pred_as_input=FLAGS.enforce_pred_as_input,
        enforce_permutations=FLAGS.enforce_permutations,
        chunk_length=FLAGS.chunk_length,
    )

    train_args = dict(
        sizes=current_algo_train_lengths,
        split='train',
        batch_size=train_batch_size,
        multiplier=-1,
        randomize_pos=FLAGS.random_pos,
        chunked=FLAGS.chunked_training,
        sampler_kwargs=sampler_kwargs,
        **common_sampler_args,
    )
    train_sampler, _, _ = make_multi_sampler(**train_args)

    mult = clrs.CLRS_30_ALGS_SETTINGS[algorithm]['num_samples_multiplier']
    val_args = dict(
        sizes=val_lengths or [np.amax(current_algo_train_lengths)],
        split='val',
        batch_size=val_batch_size,
        multiplier=2 * mult,
        randomize_pos=FLAGS.random_pos,
        chunked=False,
        sampler_kwargs=sampler_kwargs,
        **common_sampler_args,
    )
    val_sampler, val_samples, _ = make_multi_sampler(**val_args)

    test_args = dict(sizes=test_lengths or [-1],
                     split='test',
                     batch_size=test_batch_size,
                     multiplier=2 * mult,
                     randomize_pos=False,
                     chunked=False,
                     sampler_kwargs={},
                     **common_sampler_args)

    test_sampler, test_samples, spec = make_multi_sampler(**test_args)

    spec_list.append(spec)
    train_samplers.append(train_sampler)
    val_samplers.append(val_sampler)
    val_sample_counts.append(val_samples)
    test_samplers.append(test_sampler)
    test_sample_counts.append(test_samples)

  return (train_samplers,
          val_samplers, val_sample_counts,
          test_samplers, test_sample_counts,
          spec_list)


def main(unused_argv):
  if FLAGS.hint_mode == 'encoded_decoded':
    encode_hints = True
    decode_hints = True
  elif FLAGS.hint_mode == 'decoded_only':
    encode_hints = False
    decode_hints = True
  elif FLAGS.hint_mode == 'none':
    encode_hints = False
    decode_hints = False
  else:
    raise ValueError('Hint mode not in {encoded_decoded, decoded_only, none}.')

  train_lengths = [int(x) for x in FLAGS.train_lengths]

  rng = np.random.RandomState(FLAGS.seed)
  rng_seed_int = int(rng.randint(2**32))  # 替代 jax.random.PRNGKey

  # Create samplers
  (
      train_samplers,
      val_samplers,
      val_sample_counts,
      test_samplers,
      test_sample_counts,
      spec_list,
  ) = create_samplers(
      rng=rng,
      train_lengths=train_lengths,
      algorithms=FLAGS.algorithms,
      val_lengths=[np.amax(train_lengths)],
      test_lengths=[-1],
      train_batch_size=FLAGS.batch_size,
  )

  processor_factory = clrs.get_processor_factory(
      FLAGS.processor_type,
      use_ln=FLAGS.use_ln,
      nb_triplet_fts=FLAGS.nb_triplet_fts,
      nb_heads=FLAGS.nb_heads,
  )
  model_params = dict(
      processor_factory=processor_factory,
      hidden_dim=FLAGS.hidden_size,
      encode_hints=encode_hints,
      decode_hints=decode_hints,
      encoder_init=FLAGS.encoder_init,
      use_lstm=FLAGS.use_lstm,
      learning_rate=FLAGS.learning_rate,
      grad_clip_max_norm=FLAGS.grad_clip_max_norm,
      checkpoint_path=FLAGS.checkpoint_path,
      freeze_processor=FLAGS.freeze_processor,
      dropout_prob=FLAGS.dropout_prob,
      hint_teacher_forcing=FLAGS.hint_teacher_forcing,
      hint_repred_mode=FLAGS.hint_repred_mode,
      nb_msg_passing_steps=FLAGS.nb_msg_passing_steps,
  )

  eval_model = clrs.models.BaselineModel(
      spec=spec_list,
      dummy_trajectory=[next(t) for t in val_samplers],
      **model_params
  )
  if FLAGS.chunked_training:
    train_model = clrs.models.BaselineModelChunked(
        spec=spec_list,
        dummy_trajectory=[next(t) for t in train_samplers],
        **model_params
    )
  else:
    train_model = eval_model

  # Training loop.
  best_score = -1.0
  current_train_items = [0] * len(FLAGS.algorithms)
  step = 0
  next_eval = 0
  val_scores = [-99999.9] * len(FLAGS.algorithms)
  length_idx = 0

  while step < FLAGS.train_steps:
    feedback_list = [next(t) for t in train_samplers]

    # Initialize model.
    if step == 0:
      all_features = [f.features for f in feedback_list]
      if FLAGS.chunked_training:
        all_length_features = [all_features] + [
            [next(t).features for t in train_samplers]
            for _ in range(len(train_lengths))]
        train_model.init(all_length_features[:-1], FLAGS.seed + 1)
      else:
        train_model.init(all_features, FLAGS.seed + 1)

    # Training step.
    for algo_idx in range(len(train_samplers)):
      feedback = feedback_list[algo_idx]
      # rng split → 简化为 seed 自增（如需可转 torch.Generator）
      seed_this = rng_seed_int
      rng_seed_int = (rng_seed_int + 1) % (2**31 - 1)

      if FLAGS.chunked_training:
        length_and_algo_idx = (length_idx, algo_idx)
      else:
        length_and_algo_idx = algo_idx

      cur_loss = train_model.feedback(seed_this, feedback, length_and_algo_idx)
      # with autocast(device_type='cuda', dtype=torch.float16):
      #     cur_loss = train_model.feedback(seed_this, feedback, length_and_algo_idx)

      if FLAGS.chunked_training:
        examples_in_chunk = np.sum(feedback.features.is_last).item()
      else:
        examples_in_chunk = len(feedback.features.lengths)
      current_train_items[algo_idx] += examples_in_chunk
      logging.info('Algo %s step %i current loss %f, current_train_items %i.',
                   FLAGS.algorithms[algo_idx], step,
                   cur_loss, current_train_items[algo_idx])

    # Periodically evaluate model
    if step >= next_eval:
      eval_model.params = train_model.params
      for algo_idx in range(len(train_samplers)):
        common_extras = {'examples_seen': current_train_items[algo_idx],
                         'step': step,
                         'algorithm': FLAGS.algorithms[algo_idx]}

        val_stats = collect_and_eval(
            val_samplers[algo_idx],
            functools.partial(eval_model.predict, algorithm_index=algo_idx),
            val_sample_counts[algo_idx],
            rng_seed_int,
            extras=common_extras)
        rng_seed_int = (rng_seed_int + 1) % (2**31 - 1)
        logging.info('(val) algo %s step %d: %s',
                     FLAGS.algorithms[algo_idx], step, val_stats)
        val_scores[algo_idx] = val_stats['score']

      next_eval += FLAGS.eval_every

      msg = (f'best avg val score was '
             f'{best_score/len(FLAGS.algorithms):.3f}, '
             f'current avg val score is {np.mean(val_scores):.3f}, '
             f'val scores are: ')
      msg += ', '.join(
          ['%s: %.3f' % (x, y) for (x, y) in zip(FLAGS.algorithms, val_scores)])
      if (sum(val_scores) > best_score) or step == 0:
        best_score = sum(val_scores)
        logging.info('Checkpointing best model, %s', msg)
        train_model.save_model('best.pkl')
      else:
        logging.info('Not saving new best model, %s', msg)

    step += 1
    length_idx = (length_idx + 1) % len(train_lengths)

  logging.info('Restoring best model from checkpoint...')
  eval_model.restore_model('best.pkl', only_load_processor=False)

  for algo_idx in range(len(train_samplers)):
    common_extras = {'examples_seen': current_train_items[algo_idx],
                     'step': step,
                     'algorithm': FLAGS.algorithms[algo_idx]}
    rng_seed_int = (rng_seed_int + 1) % (2 ** 31 - 1)

    test_stats = collect_and_eval(
        test_samplers[algo_idx],
        functools.partial(eval_model.predict, algorithm_index=algo_idx),
        test_sample_counts[algo_idx],
        rng_seed_int,
        extras=common_extras)

    logging.info('(test) algo %s : %s', FLAGS.algorithms[algo_idx], test_stats)
    print('(test) algo %s : %s', FLAGS.algorithms[algo_idx], test_stats)
  logging.info('Done!')


if __name__ == '__main__':
  app.run(main)
