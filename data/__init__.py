from data.four_modal_dataset import Four_modal_DataSet
from data.target_4modal_dataset import target_4modal

def get_loader_4modal(name):
    """get_loader
    :param name:
    """
    return {
        "BraTs": Four_modal_DataSet,
        "tongji": target_4modal,
        "ruijin": target_4modal,
        "huashan": target_4modal,
        "xinhua": target_4modal,
        "FeTS15": target_4modal,
        "FeTS16": target_4modal,
        "FeTS17": target_4modal
    }[name]

def get_data_path_4modal(name):
    """get_data_path
    :param name:
    :param config_file:
    """

    if name == 'BraTs':
        return './dataset/BraTs/'
    if name == 'ruijin':
        return './dataset/ruijin/'
    if name == 'huashan':
        return './dataset/huashan/'
    if name == 'xinhua':
        return './dataset/xinhua/'
    if name == 'tongji':
        return './dataset/tongji/'
    if name == 'FeTS15':
        return './dataset/FeTS15/'
    if name == 'FeTS16':
        return './dataset/FeTS16/'
    if name == 'FeTS17':
        return './dataset/FeTS17/'
