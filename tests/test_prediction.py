import torch

from prismatic.models.vlas.prediction import LatentPredictor


def test_latent_predictor_shapes():
    batch = 2
    context_dim = 2048
    latent_dim = 256
    llm_dim = 1024

    predictor = LatentPredictor(context_dim=context_dim, latent_dim=latent_dim, llm_embed_dim=llm_dim)
    x = torch.randn(batch, context_dim)
    latent, latent_llm = predictor(x)

    assert latent.shape == (batch, latent_dim)
    assert latent_llm.shape == (batch, llm_dim)

    # test encode target from patches
    patches = torch.randn(batch, 50, predictor.mlp[-1].out_features if hasattr(predictor.mlp[-1], 'out_features') else 512)
    tgt = predictor.encode_target_from_patches(patches)
    assert tgt.shape[0] == batch
