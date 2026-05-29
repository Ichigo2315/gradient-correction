import importlib
import sys

sys.path.append('.')
sys.path.append('..')

import torch
import torch.multiprocessing as mp

from networks.managers.evaluator import Evaluator


def main_worker(gpu, cfg, seq_queue=None, info_queue=None, enable_amp=False):
    # Initiate a evaluating manager
    evaluator = Evaluator(rank=gpu,
                          cfg=cfg,
                          seq_queue=seq_queue,
                          info_queue=info_queue)
    # Start evaluation
    if enable_amp:
        with torch.cuda.amp.autocast(enabled=True):
            evaluator.evaluating()
    else:
        evaluator.evaluating()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Eval VOS")
    parser.add_argument('--exp_name', type=str, default='default')

    parser.add_argument('--stage', type=str, default='pre')
    parser.add_argument('--model', type=str, default='aott')
    parser.add_argument('--lstt_num', type=int, default=-1)
    parser.add_argument('--lt_gap', type=int, default=-1)
    parser.add_argument('--st_skip', type=int, default=-1)
    parser.add_argument('--max_id_num', type=int, default='-1')

    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--gpu_num', type=int, default=1)

    parser.add_argument('--ckpt_path', type=str, default='')
    parser.add_argument('--ckpt_step', type=int, default=-1)

    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--split', type=str, default='')

    parser.add_argument('--ema', action='store_true')
    parser.set_defaults(ema=False)

    parser.add_argument('--flip', action='store_true')
    parser.set_defaults(flip=False)
    parser.add_argument('--ms', nargs='+', type=float, default=[1.])

    parser.add_argument('--max_resolution', type=float, default=480 * 1.3)

    parser.add_argument('--amp', action='store_true')
    parser.set_defaults(amp=False)

    # Gradient correction toggle (default: enabled). Use --no_gc for baseline.
    parser.add_argument('--no_gc', dest='gradient_correction',
                        action='store_false',
                        help='disable test-time gradient correction '
                             '(clean no-GC baseline)')
    parser.set_defaults(gradient_correction=True)
    parser.add_argument('--gc_rate', type=float, default=-1,
                        help='gradient-correction step size alpha')
    parser.add_argument('--gc_interval', type=int, default=-1,
                        help='gradient-correction frame interval K')
    parser.add_argument('--gc_iter', type=int, default=-1,
                        help='gradient-correction inner steps N')
    parser.add_argument('--gc_debug', action='store_true',
                        help='print per-frame GC gradient diagnostics')
    parser.set_defaults(gc_debug=False)
    parser.add_argument('--gc_adaptive', action='store_true',
                        help='uncertainty-gated adaptive step size '
                             '(freeze confident pixels)')
    parser.set_defaults(gc_adaptive=False)
    parser.add_argument('--gc_adapt_gamma', type=float, default=-1,
                        help='entropy-weight sharpening exponent for adaptive GC')
    parser.add_argument('--gc_legacy', action='store_true',
                        help='use the original (legacy) GC implementation')
    parser.set_defaults(gc_legacy=False)

    args = parser.parse_args()

    engine_config = importlib.import_module('configs.' + args.stage)
    cfg = engine_config.EngineConfig(args.exp_name, args.model)

    cfg.TEST_EMA = args.ema

    cfg.TEST_GPU_ID = args.gpu_id
    cfg.TEST_GPU_NUM = args.gpu_num

    if args.lstt_num > 0:
        cfg.MODEL_LSTT_NUM = args.lstt_num
    if args.lt_gap > 0:
        cfg.TEST_LONG_TERM_MEM_GAP = args.lt_gap
    if args.st_skip > 0:
        cfg.TEST_SHORT_TERM_MEM_SKIP = args.st_skip

    if args.max_id_num > 0:
        cfg.MODEL_MAX_OBJ_NUM = args.max_id_num

    if args.ckpt_path != '':
        cfg.TEST_CKPT_PATH = args.ckpt_path
    if args.ckpt_step > 0:
        cfg.TEST_CKPT_STEP = args.ckpt_step

    if args.dataset != '':
        cfg.TEST_DATASET = args.dataset

    if args.split != '':
        cfg.TEST_DATASET_SPLIT = args.split

    cfg.TEST_FLIP = args.flip
    cfg.TEST_MULTISCALE = args.ms

    cfg.TEST_GRADIENT_CORRECTION = args.gradient_correction
    cfg.TEST_GC_DEBUG = args.gc_debug
    cfg.TEST_GC_ADAPTIVE = args.gc_adaptive
    cfg.TEST_GC_LEGACY = args.gc_legacy
    if args.gc_rate > 0:
        cfg.TEST_GC_RATE = args.gc_rate
    if args.gc_interval > 0:
        cfg.TEST_GC_INTERVAL = args.gc_interval
    if args.gc_iter > 0:
        cfg.TEST_GC_ITER = args.gc_iter
    if args.gc_adapt_gamma > 0:
        cfg.TEST_GC_ADAPT_GAMMA = args.gc_adapt_gamma

    if cfg.TEST_MULTISCALE != [1.]:
        cfg.TEST_MAX_SHORT_EDGE = args.max_resolution  # for preventing OOM
    else:
        cfg.TEST_MAX_SHORT_EDGE = None  # the default resolution setting of CFBI and AOT
    cfg.TEST_MAX_LONG_EDGE = args.max_resolution * 800. / 480.

    if args.gpu_num > 1:
        mp.set_start_method('spawn')
        seq_queue = mp.Queue()
        info_queue = mp.Queue()
        mp.spawn(main_worker,
                 nprocs=cfg.TEST_GPU_NUM,
                 args=(cfg, seq_queue, info_queue, args.amp))
    else:
        main_worker(0, cfg, enable_amp=args.amp)


if __name__ == '__main__':
    main()
