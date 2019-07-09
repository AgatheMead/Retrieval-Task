from .util import Dataset
from .dch_s3d import DCH_S3D

def train(train_img, database_img, query_img, config):
    '''train_img belongs to object Kinetics defined in data_provider/img/__init__.py '''
    #model = DCH_Inception(config)
    model = DCH_S3D(config)
    #print('trainning set object class loaded')
    img_database = Dataset(database_img, config.output_dim)
    img_query = Dataset(query_img, config.output_dim)
    img_train = Dataset(train_img, config.output_dim)
    model.train(img_train)
    return model.save_file

def validation(database_img, query_img, config):
    #model = DCH_Inception(config)
    model = DCH_S3D(config)
    img_database = Dataset(database_img, config.output_dim)
    img_query = Dataset(query_img, config.output_dim)
    return model.validation(img_query, img_database, config.R)
