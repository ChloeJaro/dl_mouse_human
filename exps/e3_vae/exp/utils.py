from omegaconf import OmegaConf


def save_config(config, path):
    with open(path, "w") as f:
        OmegaConf.save(config=config, f=f.name)
