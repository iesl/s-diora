import os
import json
import sys


def read_flags(fn):
    with open(fn) as f:
        flags = json.loads(f.read())
    return flags


def override_with_flags(options, flags, flags_to_use=None):
    """
    If `flags_to_use` is None, then override all flags, otherwise,
    only consider flags from `flags_to_use`.

    """
    if flags_to_use is None:
        for k, v in flags.items():
            setattr(options, k, v)
    else:
        for k in flags_to_use:
            default_val = options.__dict__.get(k, None)
            setattr(options, k, flags.get(k, default_val))
    return options


def init_with_flags_file(options, flags_file, flags_to_use=None):
    flags = read_flags(flags_file)
    options = override_with_flags(options, flags, flags_to_use)
    return options


def init_boolean_flags(options, other_args):
    raise Exception('Deprecated.')

    flags_to_add = {}

    # Add a 'no'-prefixed arg for all boolean args.
    for k, v in options.__dict__.items():
        if isinstance(v, bool):
            if k.startswith('no'):
                flags_to_add[k[2:]] = not v
            else:
                flags_to_add['no' + k] = not v

    for k, v in flags_to_add.items():
        setattr(options, k, v)

    # Handled 'no'-prefixed args that were not explicitly defined.
    for arg in other_args:
        if arg.startswith('--'):
            arg = arg[2:]

            # Set boolean arg to False
            if arg.startswith('no'):
                arg = arg[2:]
                if hasattr(options, arg) and isinstance(options.__dict__[arg], bool):
                    options.__dict__[arg] = False
                    options.__dict__['no' + arg] = True

            # Set boolean arg to True
            else:
                if hasattr(options, arg) and isinstance(options.__dict__[arg], bool):
                    options.__dict__[arg] = True
                    options.__dict__['no' + arg] = False

    return options


def stringify_flags(options):
    # Ignore negative boolean flags.
    flags = {k: v for k, v in options.__dict__.items()}
    return json.dumps(flags, indent=4, sort_keys=True)


def save_flags(options, experiment_path):
    flags = stringify_flags(options)
    target_file = os.path.join(experiment_path, 'flags.json')
    with open(target_file, 'w') as f:
        f.write(flags)
