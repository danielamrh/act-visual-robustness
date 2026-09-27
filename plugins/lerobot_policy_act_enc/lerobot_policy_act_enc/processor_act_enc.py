from lerobot.policies.act.processor_act import make_act_pre_post_processors


def make_act_enc_pre_post_processors(config, dataset_stats=None):
    """Same pre/post-processing as ACT (normalization, device, batching)."""
    return make_act_pre_post_processors(config, dataset_stats=dataset_stats)
