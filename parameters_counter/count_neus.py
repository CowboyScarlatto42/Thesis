#!/usr/bin/env python3
"""
Count parameters of main NeuS components and print results.

Usage:
  python3 count_neus.py [--conf path/to/conf.conf]

Output (stdout):
  ElementName : <number>
  ...
  Total : <number>

The script expects the `NeuS` package to be importable (e.g. set PYTHONPATH to
the parent folder that contains the `NeuS` package or create a symlink).

If --conf is provided and `pyhocon` is installed, the script will try to read
top-level keys `sdf_network`, `rendering_network`, `nerf`, `variance_network`
and use their dict values as init args for the corresponding classes.
Otherwise default init args are used (matching typical thin_structure.conf).
"""
import argparse
import json
import sys
from types import SimpleNamespace

def safe_import(name):
    try:
        module = __import__(name, fromlist=['*'])
        return module
    except Exception as e:
        print(f"ERROR: cannot import module '{name}': {e}", file=sys.stderr)
        return None

def load_hocon(path):
    try:
        from pyhocon import ConfigFactory
    except Exception:
        print("pyhocon not installed; ignoring --conf", file=sys.stderr)
        return {}
    try:
        cfg = ConfigFactory.parse_file(path)
        return dict(cfg)
    except Exception as e:
        print(f"Failed to parse config '{path}': {e}", file=sys.stderr)
        return {}

def instantiate_class(module, class_name, init_args):
    if module is None:
        raise ImportError(f"module is None for class {class_name}")
    cls = getattr(module, class_name, None)
    if cls is None:
        raise AttributeError(f"class {class_name} not found in module {module.__name__}")
    try:
        return cls(**init_args)
    except TypeError as e:
        # show a helpful message and try without args
        print(f"Warning: failed to instantiate {class_name} with args {init_args}: {e}", file=sys.stderr)
        try:
            return cls()
        except Exception as e2:
            raise

def count_params(model):
    import torch
    total = 0
    trainable = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return total, trainable

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--conf', help='Optional HOCON config file to override init args', default=None)
    args = p.parse_args()

    cfg = {}
    if args.conf:
        cfg = load_hocon(args.conf)

    # import NeuS model definitions
    neu_module = safe_import('NeuS.models.fields')
    if neu_module is None:
        sys.exit(2)

    # default init-args (taken from thin_structure.conf conventions)
    defaults = {
        'SDFNetwork': {"d_in":3,"d_out":257,"d_hidden":256,"n_layers":8,"skip_in":[4],"multires":6,"bias":0.5,"scale":3.0,"geometric_init":True,"weight_norm":True},
        'RenderingNetwork': {"d_feature":256,"mode":"idr","d_in":9,"d_out":3,"d_hidden":256,"n_layers":4,"weight_norm":True,"squeeze_out":True},
        'NeRF': {"D":8,"W":256,"d_in":4,"d_in_view":3,"multires":10,"output_ch":4,"skips":[4],"use_viewdirs":True},
        'SingleVarianceNetwork': {"init_val":0.3},
    }

    # allow config overrides: accept either top-level keys or nested under 'model'
    overrides = {}
    if cfg:
        for key in ['sdf_network','rendering_network','nerf','variance_network','model']:
            if key in cfg and isinstance(cfg[key], dict):
                # map to our class names
                if key == 'sdf_network':
                    overrides['SDFNetwork'] = dict(cfg[key])
                elif key == 'rendering_network':
                    overrides['RenderingNetwork'] = dict(cfg[key])
                elif key == 'nerf':
                    overrides['NeRF'] = dict(cfg[key])
                elif key == 'variance_network':
                    overrides['SingleVarianceNetwork'] = dict(cfg[key])
                elif key == 'model':
                    # try common nested names
                    for sub in ('sdf_network','rendering_network','nerf','variance_network'):
                        if sub in cfg['model']:
                            if sub == 'sdf_network': overrides['SDFNetwork'] = dict(cfg['model'][sub])
                            if sub == 'rendering_network': overrides['RenderingNetwork'] = dict(cfg['model'][sub])
                            if sub == 'nerf': overrides['NeRF'] = dict(cfg['model'][sub])
                            if sub == 'variance_network': overrides['SingleVarianceNetwork'] = dict(cfg['model'][sub])

    components = [
        ('SDFNetwork','SDFNetwork'),
        ('RenderingNetwork','RenderingNetwork'),
        ('NeRF','NeRF'),
        ('SingleVarianceNetwork','SingleVarianceNetwork'),
    ]

    results = []
    total_sum = 0

    # suppress warnings and avoid extra prints — we only want counts on stdout
    try:
        import warnings
        warnings.filterwarnings("ignore")
        import torch  # may emit FutureWarning; suppressed above
    except Exception:
        # If torch isn't available, counting will fail below; we don't print here
        pass

    for display_name, class_name in components:
        init_args = defaults.get(class_name, {}).copy()
        if class_name in overrides:
            # merge overrides
            try:
                init_args.update(overrides[class_name])
            except Exception:
                pass

        try:
            model = instantiate_class(neu_module, class_name, init_args)
        except Exception as e:
            print(f"Failed to create {class_name}: {e}", file=sys.stderr)
            results.append((display_name, 0))
            continue

        try:
            total, trainable = count_params(model)
        except Exception as e:
            print(f"Failed to count params for {class_name}: {e}", file=sys.stderr)
            results.append((display_name, 0))
            continue

        results.append((display_name, int(total)))
        total_sum += int(total)

    # print results in requested format
    for name, val in results:
        print(f"{name} : {val}")
    print(f"Total : {total_sum}")

if __name__ == '__main__':
    main()
