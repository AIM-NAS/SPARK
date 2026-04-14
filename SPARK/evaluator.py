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

import functools
import inspect
import multiprocessing as mp
import os
import shutil
import sys
from typing import Any, Dict, List, Optional


def _mp_start_method() -> Optional[str]:
  try:
    return mp.get_start_method(allow_none=True)
  except TypeError:
    try:
      return mp.get_start_method()
    except RuntimeError:
      return None
  except RuntimeError:
    return None


def _truthy_env(name: str, default: str = '0') -> bool:
  return os.environ.get(name, default).strip().lower() not in ('', '0', 'false', 'no', 'off')


def _force_cpu_in_this_process() -> bool:
  return _mp_start_method() == 'fork' and _truthy_env('OE_FORCE_CPU_IN_FORK', '1')


if _force_cpu_in_this_process():
  os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
  os.environ['OE_EVAL_DEVICE'] = 'cpu'

import torch

if _force_cpu_in_this_process():
  torch.cuda.is_available = lambda: False

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from openevolve.evaluation_result import EvaluationResult

from absl import app
from absl import flags
from absl import logging
import numpy as np
import requests


def _configure_torch_runtime() -> None:
  torch.set_float32_matmul_precision('high')
  if not _force_cpu_in_this_process() and torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True


def _device_mode() -> str:
  if _force_cpu_in_this_process():
    return 'cpu'
  return 'cuda' if torch.cuda.is_available() else 'cpu'


FLAGS = flags.FLAGS

global_triple_path = '/data1/lz/clrs/openevolve/clrs_pytorch_evolve/initial_program.py'
def set_global_triple_path(path: str):
    """更新本轮评测使用的 initial_program.py 路径"""
    global global_triple_path
    global_triple_path = path
# ===== 追加：参数量统计 & 推理时延测量（与 evaluator_evolve 对齐） =====
def _count_params_py(params) -> int:
  """统计 PyTorch/CLRS pytorch params 的总参数量。"""
  try:
    # 你的 BaselineModel.params 是“JAX 风格树” or Python 容器；这里做鲁棒统计
    import numpy as np
    def _cnt(x):
      try:
        s = x.shape
        n = 1
        for d in s: n *= int(d)
        return int(n)
      except Exception:
        try:
          return int(np.prod(getattr(x, 'shape', ())))
        except Exception:
          return 0
    if isinstance(params, dict):
      return sum(_cnt(v) for v in params.values())
    if isinstance(params, (list, tuple)):
      return sum(_cnt(v) for v in params)
    return _cnt(params)
  except Exception:
    return 0


def _measure_predict_median_pt(eval_model, algo_name: str, seed_int: int,
                               algorithm_index: int = 0,
                               batch_size: int = 16, runs: int = 5, warmup: int = 2,
                               length: int = 4) -> float:
  import time
  sampler, spec = clrs.build_sampler(algo_name, seed=12345, num_samples=-1, length=length)
  def _it():
    while True:
      yield sampler.next(batch_size)
  it = _it()
  fb = next(it)  # 预热初始化一次

  # —— 关键：传 algorithm_index，并且所有 predict 前确保模型已 init ——
  _ = eval_model.predict(seed_int, fb.features, algorithm_index=algorithm_index)

  for _ in range(warmup):
    fb = next(it)
    _ = eval_model.predict(seed_int, fb.features, algorithm_index=algorithm_index)

  times = []
  for _ in range(runs):
    fb = next(it)
    t0 = time.perf_counter()
    _ = eval_model.predict(seed_int, fb.features, algorithm_index=algorithm_index)
    times.append(time.perf_counter() - t0)
  times.sort()
  return float(times[len(times)//2])



# === 关键替换：使用 clrs_pytorch 而非 clrs / haiku / jax ===
import clrs_pytorch as clrs



# -------------------- Flags（保持与原版一致） --------------------
if 'algorithms' not in FLAGS:
    flags.DEFINE_list('algorithms', ['dfs'], 'Which algorithms to run.')

if 'train_lengths' not in FLAGS:
    flags.DEFINE_list(
        'train_lengths', ['11'],
        'Which training sizes to use. A size of -1 means use the benchmark dataset.'
    )

if 'length_needle' not in FLAGS:
    flags.DEFINE_integer(
        'length_needle', -8,
        'Length of needle for train/val in string matching algos. '
        'Negative=random(1..abs(val)); 0=use 1/4 of haystack.'
    )

if 'seed' not in FLAGS:
    flags.DEFINE_integer('seed', 42, 'Random seed to set')

if 'random_pos' not in FLAGS:
    flags.DEFINE_boolean('random_pos', True, 'Randomize the pos input common to all algos.')

if 'enforce_permutations' not in FLAGS:
    flags.DEFINE_boolean('enforce_permutations', True, 'Whether to enforce permutation-type node pointers.')

if 'enforce_pred_as_input' not in FLAGS:
    flags.DEFINE_boolean('enforce_pred_as_input', True, 'Whether to change pred_h hints into pred inputs.')

if 'batch_size' not in FLAGS:
    flags.DEFINE_integer('batch_size', 32, 'Batch size used for training.')

if 'chunked_training' not in FLAGS:
    flags.DEFINE_boolean('chunked_training', False, 'Whether to use chunking for training.')

if 'chunk_length' not in FLAGS:
    flags.DEFINE_integer('chunk_length', 16, 'Time chunk length used for training (if chunked).')

if 'train_steps' not in FLAGS:
    flags.DEFINE_integer('train_steps', 300, 'Number of training iterations.')

if 'eval_every' not in FLAGS:
    flags.DEFINE_integer('eval_every', 150, 'Evaluation frequency (in steps).')

if 'test_every' not in FLAGS:
    flags.DEFINE_integer('test_every', 300, 'Test frequency (in steps).')

if 'hidden_size' not in FLAGS:
    flags.DEFINE_integer('hidden_size', 128, 'Number of hidden units of the model.')

if 'nb_heads' not in FLAGS:
    flags.DEFINE_integer('nb_heads', 1, 'Number of heads for GAT processors.')

if 'nb_msg_passing_steps' not in FLAGS:
    flags.DEFINE_integer('nb_msg_passing_steps', 1, 'Number of message passing steps per hint.')

if 'learning_rate' not in FLAGS:
    flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate to use.')

if 'grad_clip_max_norm' not in FLAGS:
    flags.DEFINE_float('grad_clip_max_norm', 1.0, 'Gradient clipping by norm. 0.0 disables.')

if 'dropout_prob' not in FLAGS:
    flags.DEFINE_float('dropout_prob', 0.0, 'Dropout rate to use.')

if 'hint_teacher_forcing' not in FLAGS:
    flags.DEFINE_float('hint_teacher_forcing', 0.0,
                       'Probability of encoding teacher hints during training (encoded_decoded modes).')

if 'hint_mode' not in FLAGS:
    flags.DEFINE_enum(
        'hint_mode', 'encoded_decoded',
        ['encoded_decoded', 'decoded_only', 'none'],
        'How hints are used.'
    )

if 'hint_repred_mode' not in FLAGS:
    flags.DEFINE_enum(
        'hint_repred_mode', 'soft',
        ['soft', 'hard', 'hard_on_eval'],
        'How to process predicted hints when fed back as inputs.'
    )

if 'use_ln' not in FLAGS:
    flags.DEFINE_boolean('use_ln', True, 'Whether to use layer norm in the processor.')

if 'use_lstm' not in FLAGS:
    flags.DEFINE_boolean('use_lstm', False, 'Whether to insert an LSTM after message passing.')

if 'nb_triplet_fts' not in FLAGS:
    flags.DEFINE_integer('nb_triplet_fts', 8, 'How many triplet features to compute.')

if 'encoder_init' not in FLAGS:
    flags.DEFINE_enum('encoder_init', 'xavier_on_scalars',
                      ['default', 'xavier_on_scalars'],
                      'Initialiser to use for the encoders.')

if 'processor_type' not in FLAGS:
    flags.DEFINE_enum(
        'processor_type', 'triplet_gmpnn',
        ['deepsets', 'mpnn', 'pgn', 'pgn_mask',
         'triplet_mpnn', 'triplet_pgn', 'triplet_pgn_mask',
         'gat', 'gatv2', 'gat_full', 'gatv2_full',
         'gpgn', 'gpgn_mask', 'gmpnn',
         'triplet_gpgn', 'triplet_gpgn_mask', 'triplet_gmpnn'],
        'Processor type to use as the network P.'
    )

if 'checkpoint_path' not in FLAGS:
    flags.DEFINE_string('checkpoint_path', '/tmp/CLRS30', 'Path to save checkpoints.')

if 'dataset_path' not in FLAGS:
    flags.DEFINE_string(
        'dataset_path',
        '/data1/lz/clrs/openevolve/dataset/CLRS30_v1.0.0.tar/CLRS30_v1.0.0/clrs_dataset',
        'Path in which dataset is stored.'
    )

if 'freeze_processor' not in FLAGS:
    flags.DEFINE_boolean('freeze_processor', False, 'Whether to freeze the processor of the model.')


def _ensure_flags_parsed():
    """若外部未调用 app.run，则用默认值快速解析一次 FLAGS。"""
    try:
        if not FLAGS.is_parsed():
            # 用一个伪 argv 触发 absl 的解析逻辑，全部使用默认值或你在 overrides 中写入的值
            FLAGS(['__clrs_pytorch_evaluator__'])
    except Exception:
        # 兜底：即使解析失败也不要阻塞（一般不会走到这里）
        pass

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

def load_initial_program(path: str):
    import importlib.util, sys
    if not os.path.isfile(path):
        raise FileNotFoundError(f"initial_program file not found: {path}")
    # —— 确保热重载：先把旧的删掉 ——
    if "evolve_mod" in sys.modules:
        del sys.modules["evolve_mod"]
    spec = importlib.util.spec_from_file_location("evolve_mod", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["evolve_mod"] = mod
    spec.loader.exec_module(mod)
    if not hasattr(mod, "get_processor_factory"):
        raise AttributeError("initial_program must define get_processor_factory(...)")
    return mod

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

  # # Create samplers
  # (
  #     train_samplers,
  #     val_samplers,
  #     val_sample_counts,
  #     test_samplers,
  #     test_sample_counts,
  #     spec_list,
  # ) = create_samplers(
  #     rng=rng,
  #     train_lengths=train_lengths,
  #     algorithms=FLAGS.algorithms,
  #     val_lengths=[np.amax(train_lengths)],
  #     test_lengths=[-1],
  #     train_batch_size=FLAGS.batch_size,
  # )
  # mod = load_initial_program(global_triple_path)
  #
  # def build_processor_factory(mod, **kwargs):
  #     import inspect
  #     sig = inspect.signature(mod.get_processor_factory)
  #     usable = {k: v for k, v in kwargs.items() if k in sig.parameters}
  #     return mod.get_processor_factory(**usable)
  #
  # processor_factory = build_processor_factory(
  #     mod,
  #     kind="mpnn",
  #     use_ln=FLAGS.use_ln,
  #     nb_triplet_fts=FLAGS.nb_triplet_fts,
  #     nb_heads=None
  # )
  #
  # model_params = dict(
  #     processor_factory=processor_factory,
  #     hidden_dim=FLAGS.hidden_size,
  #     encode_hints=encode_hints,
  #     decode_hints=decode_hints,
  #     encoder_init=FLAGS.encoder_init,
  #     use_lstm=FLAGS.use_lstm,
  #     learning_rate=FLAGS.learning_rate,
  #     grad_clip_max_norm=FLAGS.grad_clip_max_norm,
  #     checkpoint_path=FLAGS.checkpoint_path,
  #     freeze_processor=FLAGS.freeze_processor,
  #     dropout_prob=FLAGS.dropout_prob,
  #     hint_teacher_forcing=FLAGS.hint_teacher_forcing,
  #     hint_repred_mode=FLAGS.hint_repred_mode,
  #     nb_msg_passing_steps=FLAGS.nb_msg_passing_steps,
  # )
  #
  # eval_model = clrs.models.BaselineModel(
  #     spec=spec_list,
  #     dummy_trajectory=[next(t) for t in val_samplers],
  #     **model_params
  # )
  #
  # if FLAGS.chunked_training:
  #   train_model = clrs.models.BaselineModelChunked(
  #       spec=spec_list,
  #       dummy_trajectory=[next(t) for t in train_samplers],
  #       **model_params
  #   )
  # else:
  #   train_model = eval_model




# ===== Stage 1：结构可用性 + 速度/规模体检（不训练） =====
def evaluate_stage1(program_path, *args, **kwargs):
  _ensure_flags_parsed()
  _configure_torch_runtime()
  try:
      # 1) 加载 initial_program，拿 processor_factory

      mod = load_initial_program(program_path)

      def build_processor_factory(mod, **kwargs):
          import inspect
          sig = inspect.signature(mod.get_processor_factory)
          usable = {k: v for k, v in kwargs.items() if k in sig.parameters}
          return mod.get_processor_factory(**usable)

      processor_factory = build_processor_factory(
          mod,
          kind="mpnn",
          use_ln=FLAGS.use_ln,
          nb_triplet_fts=FLAGS.nb_triplet_fts,
          nb_heads=None
      )

      # 2) 复用你的 create_samplers 逻辑，构一个 eval_model（不需要 train_model）
      train_lengths = [int(x) for x in FLAGS.train_lengths]
      rng = np.random.RandomState(FLAGS.seed)

      (train_samplers,
       val_samplers,
       val_sample_counts,
       test_samplers,
       test_sample_counts,
       spec_list) = create_samplers(
          rng=rng,
          train_lengths=train_lengths,
          algorithms=FLAGS.algorithms,
          val_lengths=[np.amax(train_lengths)],
          test_lengths=[-1],
          train_batch_size=FLAGS.batch_size,
      )

      # 3) 构建未训练的 eval_model，并用一小批 features 做 init
      encode_hints = FLAGS.hint_mode in ('encoded_decoded',)
      decode_hints = FLAGS.hint_mode in ('encoded_decoded', 'decoded_only')

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
      # —— 关键：像训练 step=0 一样做一次 init ——
      init_features = [next(t).features for t in val_samplers]
      eval_model.init(init_features, FLAGS.seed + 1)

      # 用最小 batch 初始化（沿用你原本的做法）
      _ = [next(t) for t in val_samplers]  # 再拉一批确保结构稳定
      # BaselineModel 的 .init 在构造时已经做过（dummy_trajectory）；这里直接统计

      # 4) 统计参数量 & 推理时延中位数
      param_count = _count_params_py(eval_model.params)
      infer_time = _measure_predict_median_pt(
          eval_model,
          FLAGS.algorithms[0],
          seed_int=FLAGS.seed + 123,
          algorithm_index=0,  # <— 新增
          batch_size=16, runs=5, warmup=2, length=4
      )

      # 5) 组合分（可按需调整权重）
      alpha_t = float(os.environ.get("OE_ALPHA_TIME", "0.005"))
      alpha_p = float(os.environ.get("OE_ALPHA_PARAM", "0.0"))
      scale_p = float(os.environ.get("OE_PARAM_SCALE", "3e5"))
      combined = - alpha_t * infer_time - alpha_p * np.log1p(param_count / max(1.0, scale_p))
      t_max = float(os.environ.get("OE_S1_TIME_MAX", "2.5"))  # 秒
      p_max = float(os.environ.get("OE_S1_PARAM_MAX", "2_000_000"))
      ok_time = (infer_time <= t_max)
      ok_param = (param_count <= p_max)
      passed = 1.0 if (ok_time and ok_param) else 0.0

      return {
          "stage": "stage1",
          "timeout": False,
          "error": 0.0,  # <—— 新增
          "stage1_passed": passed,  # <—— 新增
          "infer_time_s": float(infer_time),
          "param_count": int(param_count),
          "macs": int(param_count),  # 这里用参数量近似 MACs
          "score_val": 0.0,
          "ood_acc": 0.0,
          "combined_score": float(combined),  # 你已有的组合分逻辑也可保留
      }
  except Exception as e:
  # 门控失败：返回 error=1、stage1_passed=0，附带错误信息，保证控制器能落地日志
    return {
      "stage": "stage1",
      "timeout": False,
      "error": 1.0,  # <—— 新增
      "stage1_passed": 0.0,  # <—— 新增
      "infer_time_s": float("inf"),
      "param_count": -1,
      "macs": -1,
      "score_val": 0.0,
      "ood_acc": 0.0,
      "combined_score": -1e9,
      "message": f"stage1-exception: {type(e).__name__}: {e}",
  }

# ===== Stage 2：完整训练-评测（等价 run.py 流程） =====
def evaluate_stage2(program_path, *args, **kwargs):
  _ensure_flags_parsed()
  _configure_torch_runtime()
    # —— 下面内容基本等同你当前 main() 的主体，只是包成函数并返回结果 —— #

  if FLAGS.hint_mode == 'encoded_decoded':
    encode_hints = True;  decode_hints = True
  elif FLAGS.hint_mode == 'decoded_only':
    encode_hints = False; decode_hints = True
  elif FLAGS.hint_mode == 'none':
    encode_hints = False; decode_hints = False
  else:
    raise ValueError('Hint mode not in {encoded_decoded, decoded_only, none}.')

  train_lengths = [int(x) for x in FLAGS.train_lengths]
  rng = np.random.RandomState(FLAGS.seed)
  rng_seed_int = int(rng.randint(2**32))

  # Create samplers（沿用你原来的）
  (train_samplers, val_samplers, val_sample_counts,
   test_samplers, test_sample_counts, spec_list) = create_samplers(
      rng=rng,
      train_lengths=train_lengths,
      algorithms=FLAGS.algorithms,
      val_lengths=[np.amax(train_lengths)],
      test_lengths=[-1],
      train_batch_size=FLAGS.batch_size,
  )

  # initial_program → processor_factory
  mod = load_initial_program(program_path)
  def build_processor_factory(mod, **kwargs):
      import inspect
      sig = inspect.signature(mod.get_processor_factory)
      usable = {k: v for k, v in kwargs.items() if k in sig.parameters}
      return mod.get_processor_factory(**usable)
  processor_factory = build_processor_factory(
      mod,
      kind="mpnn",
      use_ln=FLAGS.use_ln,
      nb_triplet_fts=FLAGS.nb_triplet_fts,
      nb_heads=None
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

  # ===== 训练循环（与你当前 main 一致） =====
  best_score = -1.0
  current_train_items = [0] * len(FLAGS.algorithms)
  step = 0
  next_eval = 0
  val_scores = [-99999.9] * len(FLAGS.algorithms)
  length_idx = 0

  import time as _time
  t0_train = _time.time()
  last_train_loss = float('nan')

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
      seed_this = rng_seed_int
      rng_seed_int = (rng_seed_int + 1) % (2**31 - 1)

      if FLAGS.chunked_training:
        length_and_algo_idx = (length_idx, algo_idx)
      else:
        length_and_algo_idx = algo_idx

      cur_loss = train_model.feedback(seed_this, feedback, length_and_algo_idx)
      last_train_loss = float(cur_loss)

      if FLAGS.chunked_training:
        examples_in_chunk = np.sum(feedback.features.is_last).item()
      else:
        examples_in_chunk = len(feedback.features.lengths)
      current_train_items[algo_idx] += examples_in_chunk
      logging.info('Algo %s step %i current loss %f, current_train_items %i.',
                   FLAGS.algorithms[algo_idx], step,
                   cur_loss, current_train_items[algo_idx])

    # Periodically evaluate
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

  train_time = _time.time() - t0_train

  # 恢复 best 并测试
  logging.info('Restoring best model from checkpoint...')
  eval_model.restore_model('best.pkl', only_load_processor=False)
  last_test = None
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
    last_test = test_stats
    logging.info('(test) algo %s : %s', FLAGS.algorithms[algo_idx], test_stats)
    print('(test) algo %s : %s', FLAGS.algorithms[algo_idx], test_stats)

  # 统计推理时延（中位）
  infer_time = _measure_predict_median_pt(
      eval_model, FLAGS.algorithms[0], seed_int=FLAGS.seed + 123,
      algorithm_index=0, batch_size=16, runs=5, warmup=2, length=4
  )
  # 组合分：以 test score 为主（可按需加惩罚）
  param_count = _count_params_py(train_model.params)
  alpha_p = float(os.environ.get("OE_ALPHA_PARAM", "0.0"))
  scale_p = float(os.environ.get("OE_PARAM_SCALE", "3e5"))
  combined = float(last_test['score']) - alpha_p * np.log1p(param_count / max(1.0, scale_p))

  return {
      "stage": "stage2",
      "timeout": False,
      "algorithm": FLAGS.algorithms[0],
      "ood_acc": float(last_test['score']),
      "last_train_loss": float(last_train_loss),
      "examples_seen": int(current_train_items[0]),
      "param_count": int(param_count),
      "macs": int(param_count),
      "nb_msg_passing_steps": int(FLAGS.nb_msg_passing_steps),
      "hidden_dim": int(FLAGS.hidden_size),
      "train_time_sec": float(train_time),
      "infer_time_s": float(infer_time),
      "test_last": last_test,
      "combined_score": float(combined),
  }




def _apply_overrides(overrides: Dict[str, Any]) -> None:
  if not overrides:
    return
  for key, value in overrides.items():
    if key in ('initial_program_path',):
      continue
    if key not in FLAGS:
      continue
    try:
      setattr(FLAGS, key, value)
    except Exception:
      pass


def evaluate(program=None, *positional, **overrides):
    """
    OpenEvolve 会 import 本文件并调用 evaluate(program, **overrides)
    - program: OpenEvolve 传入的 initial_program.py 路径（字符串）
    - overrides: OpenEvolve 传入的一些超参覆盖（字典），例如：
        algorithms='dfs' 或 ['dfs']
        train_lengths=[4, 7, 11, 13, 16]
        batch_size=32, train_steps=10000, eval_every=200, ...
    返回值: dict，至少包含 'score'（建议也带上 'combined_score'）
    """
    _ensure_flags_parsed()
    _configure_torch_runtime()
    try:
        _apply_overrides(overrides)

        program_path = None
        if isinstance(program, str) and program.strip():
            program_path = program.strip()
        elif isinstance(overrides.get('initial_program_path', ''), str) and overrides['initial_program_path'].strip():
            program_path = overrides['initial_program_path'].strip()
        elif os.environ.get('OE_INITIAL_PROGRAM_PATH', '').strip():
            program_path = os.environ['OE_INITIAL_PROGRAM_PATH'].strip()
        else:
            program_path = "/data1/lz/clrs/openevolve/clrs_pytorch_evolve/initial_program.py"

        logging.info(f"[Evaluator] Using program_path = {program_path}")
        logging.info(f"[Evaluator] device_mode = {_device_mode()}, mp_start_method = {_mp_start_method()}")
        print("program_path:", program_path)
        print("device_mode:", _device_mode(), "mp_start_method:", _mp_start_method())

        s1 = evaluate_stage1(program_path)
        if not isinstance(s1, dict) or s1.get('stage1_passed', 0.0) <= 0.0:
            out = s1 if isinstance(s1, dict) else {
                'combined_score': -1e9,
                'score': -1e9,
                'message': 'stage1 returned invalid result',
                'stage': 'stage1',
                'timeout': False,
                'ood_acc': 0.0,
            }
            out['cascade_debug'] = {'stage1': s1, 'stage2': None}
        else:
            s2 = evaluate_stage2(program_path)
            out = s2 if (isinstance(s2, dict) and not s2.get('timeout', False)) else s1
            if 'score' not in out and 'combined_score' in out:
                out['score'] = float(out['combined_score'])
            out['cascade_debug'] = {'stage1': s1, 'stage2': s2}

        if 'score' not in out and 'combined_score' in out:
            out['score'] = float(out['combined_score'])

        artifacts_dir = os.environ.get('OE_ARTIFACTS_DIR', '')
        if artifacts_dir:
            try:
                import json, pathlib
                pathlib.Path(artifacts_dir).mkdir(parents=True, exist_ok=True)
                with open(os.path.join(artifacts_dir, 'result.json'), 'w') as f:
                    json.dump(out, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

        return out

    except Exception as e:
        return {
            'combined_score': -1e9,
            'score': -1e9,
            'message': f'{type(e).__name__}: {e}',
            'stage': 'evaluate-failed',
            'timeout': False,
            'ood_acc': -1e9
        }


# def main_two_stages(_):
#   """先跑 Stage1，再跑 Stage2，并把结果打印出来。"""
#   s1 = evaluate_stage1()
#   logging.info('[STAGE1] %s', s1)
#   print('[STAGE1]', s1)
#   s2 = evaluate_stage2()
#   logging.info('[STAGE2] %s', s2)
#   print('[STAGE2]', s2)
#   # 你也可以在这里做一个 merge/挑选（默认以 Stage2 为准）
#   return 0

# if __name__ == '__main__':
  # 方式一：仍然使用原来的单阶段训练入口（向后兼容）
  # app.run(main)

  # 方式二：使用两阶段入口（推荐）
  # app.run(main_two_stages)


