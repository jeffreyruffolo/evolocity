from ..preprocessing import get_model


def velocity_model(
    adata,
    model_name='esm1b',
    model_path=None,
    vocab_file=None,
    mkey='model',
    copy=False,
):
    adata = adata.copy() if copy else adata

    model = get_model(model_name, model_path, vocab_file)

    adata.uns[mkey] = model

    return adata if copy else None
