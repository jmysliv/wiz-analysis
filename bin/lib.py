import yaml

CONFS = None


def load_confs(confs_path='conf/conf.yaml'):
    global CONFS
    if CONFS is None:
        with open(confs_path, "r") as stream:
            try:
                CONFS = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

    return CONFS


def get_conf(conf_name):
    """
    Get a configuration parameter by its name
    :param conf_name: Name of a configuration parameter
    :type conf_name: str
    :return: Value for that conf (no specific type information available)
    """
    return load_confs()[conf_name]