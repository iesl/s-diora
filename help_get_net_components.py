from input_layer import *
from help_get_diora import get_diora
from help_get_loss_funcs import get_loss_funcs


def check_params(source, target):
    source_params = {n: p for n, p in source.named_parameters()}
    target_params = {n: p for n, p in target.named_parameters()}
    # assert len(source_params) == len(target_params)

    for n, p in source_params.items():
        if not nested_hasattr(target, n):
            print('skipping check {}'.format(n))
            continue
        assert torch.all(p == target_params[n])


def nested_getattr(o, k):
    k_lst = k.split('.')
    for i in range(len(k_lst)):
        o = getattr(o, k_lst[i])
    return o


def nested_hasattr(o, k):
    try:
        _ = nested_getattr(o, k)
        return True
    except:
        return False


def nested_setattr(o, k, v):
    k_lst = k.split('.')
    if len(k_lst) > 1:
        new_k = '.'.join(k_lst[:-1])
        o = nested_getattr(o, new_k)
    setattr(o, k_lst[-1], v)


def copy_params(source, target):
    for n, p in source.named_parameters():
        if not nested_hasattr(target, n):
            print('skipping {}'.format(n))
            continue
        print('copy {} {}'.format(n, p.shape))
        nested_setattr(target, n, p)


def get_net_components(options, context):
    #
    embeddings = context['embeddings']
    word2idx = context['word2idx']
    batch_iterator = context['batch_iterator']

    # TODO: There should be a better way to do this?
    options.input_dim = embeddings.shape[1]
    if options.projection == 'word2vec':
        options.input_dim = embeddings.shape[1]
    elif options.projection == 'elmo':
        options.input_dim = 1024
    elif options.projection == 'bert':
        options.input_dim = 768
    elif options.projection == 'mask':
        raise NotImplementedError
        options.input_dim = embeddings.shape[1]

    # Embed
    projection_layer, embedding_layer = get_embed_and_project(options, embeddings, options.input_dim, options.hidden_dim, word2idx)

    # Diora
    diora_context = {}
    diora_context['projection_layer'] = projection_layer
    diora = get_diora(options, diora_context, config=options.model_config)

    # Loss
    loss_context = {}
    loss_context['batch_iterator'] = batch_iterator
    loss_context['embedding_layer'] = embedding_layer
    loss_context['diora'] = diora
    loss_context['word2idx'] = word2idx
    loss_context['embeddings'] = embeddings
    loss_context['cuda'] = options.cuda
    loss_context['input_dim'] = options.input_dim
    loss_context['projection_layer'] = projection_layer
    loss_funcs = get_loss_funcs(options, loss_context, config_lst=options.loss_config)

    # Return components.
    components = {}
    components['projection_layer'] = projection_layer
    components['embedding_layer'] = embedding_layer
    components['diora'] = diora
    components['loss_funcs'] = loss_funcs
    return components

