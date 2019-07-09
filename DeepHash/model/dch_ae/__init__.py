from .util import Dataset
from .dch_ae import DCH_AE

def train(train_img, database_img, query_img, config):
    model = DCH_AE(config)
    img_database = Dataset(database_img, config.output_dim)
    img_query = Dataset(query_img, config.output_dim)
    img_train = Dataset(train_img, config.output_dim)
    model.train(img_train)
    return model.save_file

def validation(database_img, query_img, config):
    model = DCH_AE(config)
    img_database = Dataset(database_img, config.output_dim)
    img_query = Dataset(query_img, config.output_dim)
    return model.validation(img_query, img_database, config.R)
