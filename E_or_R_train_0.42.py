import os
import time
import datetime
import pathlib
import sys
import shutil

import re # regular expressions

# import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning) #!
import pandas as pd
pd.set_option('display.max_rows', 100)

import statistics
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
#import seaborn as sns
#from labellines import labelLine, labelLines
# import pydot
import math
import tensorflow as tf
import tensorflow_addons as tfa

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sklearn.preprocessing
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# from K.utils.vis_utils import plot_model
#from keras.utils.generic_utils import get_custom_objects
from tensorflow.keras.layers import Activation
import threading
import concurrent.futures
import multiprocessing, logging
from multiprocessing import freeze_support

import configparser

import secrets

#mpl = multiprocessing.log_to_stderr()
#mpl.setLevel(logging.INFO)
#print(tf.version.VERSION)
#print(tfa.__version__)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


#==============================================================================
if __name__ == '__main__':   
    freeze_support()
    t_begin = datetime.datetime.now()
    
    
#================================
# чтение различных параметров из конфигурационного файла
cg = configparser.ConfigParser()
cg.read(pathlib.Path.cwd() / '..' / 'config.ini', encoding='utf8')

Notes = cg['Notes']['notes']

#================================    
# пути к исходным данным 
paths_to_the_data = cg['Paths to the data']    
path_data_e     = pathlib.Path(paths_to_the_data['path_energy'])    
path_data_r     = pathlib.Path(paths_to_the_data['path_radius'])
columns_data_e  = paths_to_the_data['columns_energy']
column_weight_e = paths_to_the_data['column_weight_energy']
columns_data_r  = paths_to_the_data['columns_radius']
column_weight_r = paths_to_the_data['column_weight_radius']


#================================
# настройки распараллеливания
parallelism = cg['Train and save parallelism']
training_threading    = parallelism.getboolean('training_threading')
training_threading_2  = parallelism.getboolean('training_threading_2')
predictions_threading = parallelism.getboolean('predictions_threading')    
number_of_threads     = parallelism.getint('number_of_threads')


#================================
# препроцессинг данных
    # логические
cut_horizontal_EandR  = cg['Data pre-processing logical parameters'].getboolean('cut_horizontal_e_and_r')    

    # для энергии
data_prep_e = cg['Data pre-processing parameters for E']
cut_on_min        = data_prep_e.getboolean('cut_on_min')
horizontal_cut    = data_prep_e.getboolean('horizontal_cut')    
minN_e            = data_prep_e.getint('min_Nmax')
maxN_e            = data_prep_e.getint('max_Nmax')
hwcutoff_l_e      = data_prep_e.getfloat('hwcutoff_left')
hwcutoff_r_e      = data_prep_e.getfloat('hwcutoff_right')
E_horizontal      = data_prep_e.getfloat('E_horizontal')
Nmax_for_metric_e = data_prep_e.getint('Nmax_for_metric')
scaler_in_min_e   = data_prep_e['scaler_in_min']
scaler_in_max_e   = data_prep_e['scaler_in_max']
scaler_out_min_e  = data_prep_e.getfloat('scaler_out_min')
scaler_out_max_e  = data_prep_e.getfloat('scaler_out_max')

if  scaler_in_min_e == 'None':
    scaler_in_min_e = None
else:
    scaler_in_min_e = data_prep_e.getfloat('scaler_in_min')    
    
if  scaler_in_max_e == 'None':
    scaler_in_max_e = None
else:
    scaler_in_max_e = data_prep_e.getfloat('scaler_in_max')
    
    #для радиуса
data_prep_r = cg['Data pre-processing parameters for R']        
minN_r            = data_prep_r.getint('min_Nmax')
maxN_r            = data_prep_r.getint('max_Nmax')
hwcutoff_l_r      = data_prep_r.getfloat('hwcutoff_left')
hwcutoff_r_r      = data_prep_r.getfloat('hwcutoff_right')
Nmax_for_metric_r = data_prep_r.getint('Nmax_for_metric')
scaler_in_min_r   = data_prep_r['scaler_in_min']
scaler_in_max_r   = data_prep_r['scaler_in_max']
scaler_out_min_r  = data_prep_r.getfloat('scaler_out_min')
scaler_out_max_r  = data_prep_r.getfloat('scaler_out_max')   

if  scaler_in_min_r == 'None':
    scaler_in_min_r = None
else:
    scaler_in_min_r = data_prep_r.getfloat('scaler_in_min') 
      
if  scaler_in_max_r == 'None':
    scaler_in_max_r = None
else:
    scaler_in_max_r = data_prep_r.getfloat('scaler_in_max')
    
    
#================================
# общие логические настройки обучения
gen_train_params = cg['General training parameters']
splitting           = gen_train_params.getboolean('splitting')
bias_usage          = gen_train_params.getboolean('bias_usage')
EXTRA_FITTING       = gen_train_params.getboolean('Extra_fitting')
SHUFFLE             = gen_train_params.getboolean('shuffle')
SHUFFLE_EXTRA       = gen_train_params.getboolean('shuffle_exta')
with_sample_weights = gen_train_params.getboolean('with_sample_weights')
WRITE_WEIGHTS       = gen_train_params.getboolean('write_weights')


#================================
# общие численные настройки обучения
gen_num_train_params = cg['General numeric training parameters']
EPOCHS_EXTRA                    = gen_num_train_params.getint('extra_epochs')
BS_EXTRA                        = gen_num_train_params.getint('batch_size_extra')
LR_EXTRA_COEF                   = gen_num_train_params.getfloat('LR_extra_coef')
CLR_NUM_CYCLES                  = gen_num_train_params.getint('CLR_NUM_CYCLES')
CLR_COEF_MAXLR                  = gen_num_train_params.getfloat('CLR_coef_maxLR')
CLR_SCALE_FN_POWER              = gen_num_train_params.getfloat('CLR_scale_fn_power')
EARLY_STOPPING_1_LOSS           = gen_num_train_params.getfloat('early_stopping_1_loss')
EARLY_STOPPING_2_PATIENCE       = gen_num_train_params.getint('early_stopping_2_patience')
EARLY_STOPPING_2_PATIENCE_EXTRA = gen_num_train_params.getint('early_stopping_2_patience_extra')
NUM_OF_EPOCH_SAMPLES_FOR_HISTS  = gen_num_train_params.getint('number_of_epoch_samples_for_weights_histograms')


#================================
# числовые настройки обучения
    # для энергии
num_train_params_e = cg['Numeric training parameters for E']
AFC_e      = num_train_params_e.getfloat('activ_func_coef')
AFC2_e     = num_train_params_e.getfloat('activ_func_coef_2')
AFB_e      = num_train_params_e.getfloat('activ_func_bias')
NE_e       = num_train_params_e.getint('num_of_epochs')
BS_e       = num_train_params_e['batch_size']
LR_e       = num_train_params_e.getfloat('learning_rate')
LRD_e      = num_train_params_e.getfloat('learning_rate_decay')
numofnns_e = num_train_params_e.getint('num_of_neural_networks')

if BS_e == 'None':
    BS_e = None
else:
    BS_e = num_train_params_e.getint('batch_size')
    
    # для радиуса    
num_train_params_r = cg['Numeric training parameters for R']
AFC_r      = num_train_params_r.getfloat('activ_func_coef')
AFC2_r     = num_train_params_r.getfloat('activ_func_coef_2')
AFB_r      = num_train_params_r.getfloat('activ_func_bias')
NE_r       = num_train_params_r.getint('num_of_epochs')
BS_r       = num_train_params_r['batch_size']
LR_r       = num_train_params_r.getfloat('learning_rate')
LRD_r      = num_train_params_r.getfloat('learning_rate_decay')
numofnns_r = num_train_params_r.getint('num_of_neural_networks')   

if BS_r == 'None':
    BS_r = None
else:
    BS_r = num_train_params_r.getint('batch_size')
    
    
#================================
# настройки получения предсказаний
making_preds_params = cg['Making predictions parameters']
n_e                   = making_preds_params.getint('max_Nmax_for_predictions_E')
n_r                   = making_preds_params.getint('max_Nmax_for_predictions_R')
NMAX_PREDICTIONS_STEP = making_preds_params.getint('Nmax_step')


#================================
# некоторые настройки Tensorflow
tf_params = cg['Tensoflow parameters']
TF_INTER_THREADS = tf_params.getint('inter_op_parallelism_threads')
TF_FLOAT_TYPE    = tf_params['floatx']
VERBOSITY        = tf_params.getint('verbosity')


#================================
# параметры Matplotlib для построения некоторых графиков во время обучения
plt_params = cg['Matplotlib parameters']
plot_format     = plt_params['plot_format']
figsize_width   = plt_params.getfloat('figure_figsize_width')
figsize_height  = plt_params.getfloat('figure_figsize_height')
axes_titlesize  = plt_params.getint('axes_titlesize')
axes_labelsize  = plt_params.getint('axes_labelsize')
xtick_labelsize = plt_params.getint('xtick_labelsize')
ytick_labelsize = plt_params.getint('ytick_labelsize')
#=================================
    

EPOCHS_FOR_SAMPLING = sorted(list(set(np.geomspace(1, min(NE_e, NE_r) - 1, num = NUM_OF_EPOCH_SAMPLES_FOR_HISTS, endpoint = True, dtype = int))))    
#---------------------------------
if __name__ == '__main__':   
    freeze_support()
    t_begin = datetime.datetime.now()
    # инициализация некоторых параметров       
    
    variational_min = 0  # определяется потом # вариационный минимум энергии в используемых данных
    
    tf.config.threading.set_inter_op_parallelism_threads(TF_INTER_THREADS)
    tf.keras.backend.set_floatx(TF_FLOAT_TYPE)  # 64 32
    
    plt.rc('figure',  figsize   = (figsize_width, figsize_height))  # setting default size of plots
    plt.rc('axes'  ,  titlesize = axes_titlesize)  # fontsize of the axes title
    plt.rc('axes'  ,  labelsize = axes_labelsize)  # fontsize of the x and y labels
    plt.rc('xtick' ,  labelsize = xtick_labelsize)  # fontsize of the tick labels
    plt.rc('ytick' ,  labelsize = ytick_labelsize)
    
    now = datetime.datetime.now() # current date and time
    year        = now.strftime("%Y")
    month       = now.strftime("%m")
    day         = now.strftime("%d")
    hour_minute = now.strftime("%H-%M")
    
    path_addition = day + "_" + month + "_" + year + '_' + hour_minute
    #script_name   = pathlib.Path(__file__).stem
        
    path = pathlib.Path.cwd()
    #path = 'G:/Мой диск/!ML/E_or_R_only/multiproc/Li/14Nmax_test_last_above0_600' # путь к папке для сохранения моделей     
    #path = 'C:/Users/shado/Desktop/ML/NNs/res/test'
    #path =    pathlib.Path('C:/Users/user/Desktop/ML/test')    
        
    #path = path / path_addition
    path_e = path / 'energy'
    path_r = path / 'radius'
    pics_path = path / 'pics' # путь для сохранения общих картинок
    pics_path_e = path / 'pics' / 'energy' # путь для сохранения картинок касательно энергии
    pics_path_r = path / 'pics' / 'radius' # касательно энергии
       
    
    # создание нужных папок
    pathlib.Path(path).mkdir(parents=True, exist_ok=True) 
    pathlib.Path(path_e).mkdir(parents=False, exist_ok=True) 
    pathlib.Path(path_r).mkdir(parents=False, exist_ok=True) 
        
    
        
    #----------------------------------------    
    class Logger(object):
        def __init__(self):
            self.terminal = sys.stdout
            self.log = open(os.path.join(path, 'output.txt'), "a")
        
        def write(self, message):
            with open (os.path.join(path, 'output.txt'), "a", encoding = 'utf-8') as self.log:            
                self.log.write(message)
            self.terminal.write(message)
              

        def flush(self):
            # this flush method is needed for python 3 compatibility.
            # this handles the flush command by doing nothing.
            # you might want to specify some extra behavior here.
            pass    

    sys.stdout = Logger()
    #----------------------------------------
    
    print('path_e = ', path_e)    
      

    
    
    
    
#==============================================================================
class new_callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}): 
    #def on_batch_end(self, epoch, logs={}):
        if(logs.get('loss') < EARLY_STOPPING_1_LOSS): # select the accuracy            
            print(f"\n\n !!! epoch: {epoch} \t loss is small enough, no further training !!!\n\n")            
            self.model.stop_training = True




         

def scale_fn(x):     
    return 1.0 / x ** CLR_SCALE_FN_POWER      
  
def get_predictions(data, models, maxN, n, scalerx, scalery):  # формируем датафрейм с предсказаниями
    print(f'\ngetting predictions of {len(models)} networks...\n')
    # data - датафрейм с данными: энергия или радиус; чтобы взять имя колонки и разные hOmega
    preds = []
    

    # t = np.empty([hwcutoff_r - hwcutoff_l + 1, 2]) # массив, который будет содержать Nmax, hOmega, для которых будут делаться предсказания
    # for i in range(hwcutoff_l, hwcutoff_r + 1):        
    #         t[i - hwcutoff_l][1] = i # типа hOmega # hw остается постоянным, меняется только Nmax

    # t = NHER[['Nmax', 'hOmega']][t['Nmax'] == 10].to_numpy()
    # print(t)
    # print(t[['Nmax', 'hOmega']][t['Nmax'] == 10])

    # последняя колонка - "Weight", поэтому
    if with_sample_weights == True:
        column_name = list(data.columns)[-2]
    else:
        column_name = list(data.columns)[-1]
    
    

    hOmegas = data['hOmega'].unique()
    hOmegas.sort()   
    
    

    #print('hOmegas = ', hOmegas)

    # t = np.empty(
    #     [len(hOmegas), 2])  # массив, который будет содержать Nmax, hOmega, для которых будут делаться предсказания
    # for i in range(len(hOmegas)):
    #     t[i][1] = hOmegas[i]  # типа hOmega # hw остается постоянным, меняется только Nmax

    # # print(t)

    # for N in range(maxN, n + 1, 2):  # цикл как бы по Nmax

    #     for i in range(len(hOmegas)):
    #         t[i][0] = N  # типа hOmega # hw остается постоянным, меняется только Nmax

    #     # print(t)

    #     for i in models:
    #         ts = scalerx.transform(t)
            
    #         ts = tf.convert_to_tensor(ts)

    #         ts_pred = i.predict(ts)

    #         predicted_value = scalery.inverse_transform(ts_pred)[:, 0]
    #         #r = scalery.inverse_transform(ts_pred)[:, 1]
    #         idx = models.index(i)
    #         for j in range(len(t)):  # разные hOmega
    #             preds.append([N, t[j][1], predicted_value[j], idx])
    
    
    arr_for_preds = []
    
    for N in range(maxN, n + 1, NMAX_PREDICTIONS_STEP):  # цикл как бы по Nmax
        for i in range(len(hOmegas)):
            arr_for_preds.append([N, hOmegas[i]])       
       
    
    arr_for_preds = np.array(arr_for_preds)    
    arr_for_preds = np.reshape(arr_for_preds, (-1,2))   
        
    for i in models:
        ts = scalerx.transform(arr_for_preds)                
        ts = tf.convert_to_tensor(ts)        
        ts_pred = i.predict(
            ts,
            batch_size=None,
            verbose=0,
            steps=None,
            callbacks=None,
            max_queue_size=10,
            workers=2,
            use_multiprocessing=False
            )

        predicted_value = scalery.inverse_transform(ts_pred)[:, 0]        
        idx = models.index(i)           
        
        
        
        ann_indeces = [] # номер сетки, столбец чтоб внести в датафрейм
        for i in range(len(arr_for_preds)):
            ann_indeces.append(idx)
         
        
        ann_indeces = np.array(ann_indeces).T
        #print(len(ann_indeces))
        
        predicted_value = np.array(predicted_value).T
        
        
        preds.extend(zip(arr_for_preds[:,0], arr_for_preds[:,1], predicted_value, ann_indeces))
           
    
    column_names = ["Nmax", "hOmega", column_name, "annid"]
    #print(len(preds))
    
    preds = pd.DataFrame(preds, columns=column_names)  # predictions
    

    return preds

def load_models(path_to_models_folder, num_of_nns, begin_idx): #вспомогательная функция для загрузки списка моделей
    #print(f'load models: path = {path_to_models_folder}')
    #print(f'load models: num_of_nns = {num_of_nns}')
    models = [] 
    print('SCALE FN(2)=', scale_fn(2))
    for i in range(begin_idx, begin_idx + num_of_nns):        
        models.append(keras.models.load_model(os.path.join(path_to_models_folder, str(i)), custom_objects={"scale_fn": scale_fn}))
        #models.append(keras.models.load_model(os.path.join(path_to_models_folder, str(i))))
        print(f'load models: model № {i} loaded')
        #models[i].summary()
    #print(len(models))
    return models
   
def get_predictions_using_path(number_of_threads, data, path_to_models_folder, num_of_nns, maxN, n, scalerx, scalery, no_threaded_loading = False):  # формируем датафрейм с предсказаниями
    print(f'\ngetting predictions of {num_of_nns} networks...\n')
    # data - датафрейм с данными: энергия или радиус; чтобы взять имя колонки и разные hOmega
    print(path_to_models_folder)
    
    preds = []
    models = []
    
    # models = load_models(path_to_models_folder, num_of_nns, 0)
    # print(f'\nget_predictions_using_path: models lenght =  {len(models)}\n')
    
    time_loading_models_begin = datetime.datetime.now() # время загрузки моделей
    
    if no_threaded_loading == True:
        print('\n loading models.... \n')
        models = load_models(path_to_models_folder, num_of_nns, 0)
        
    else:        
        #--------------------------------------------------------------------------
        print('\n loading models in parallel.... \n')
        #number_of_threads = multiprocessing.cpu_count() # определение числа потоков
        per_thread = num_of_nns // number_of_threads # число сетей, которое будет приходиться на один поток (не считая остатка)
            
        threads = [] # список для хранения потоков    
        threading_results = [] # список для хранения результатов   
           
        with concurrent.futures.ThreadPoolExecutor() as executor:  
            for i in range(number_of_threads - 1): # инициализация(?) потоков, кроме последнего
                # submit() method schedules each task                 
                threads.append(executor.submit(load_models, path_to_models_folder, per_thread, per_thread * i))       
            
                      
            #последний поток включает  per_thread + остаток от деления сетей (если нацело не делится)
            threads.append(executor.submit(load_models, path_to_models_folder, per_thread + num_of_nns % number_of_threads, (number_of_threads-1)*per_thread))
                                
            for i in range(number_of_threads):
                threading_results.append(threads[i].result())
                
        #объединение
        for threading_result in threading_results: 
            models.extend(threading_result)    
        #--------------------------------------------------------------------------
    time_loading_models_end = datetime.datetime.now()
    loading_models_time     = time_loading_models_end - time_loading_models_begin
    
    print(f'Loading models finished in {(time_loading_models_end - time_loading_models_begin).total_seconds()} second(s)') 
    
    print(f'\nget_predictions_using_path: models length =  {len(models)}\n')
    
    # последняя колонка - "Weight", поэтому
    if with_sample_weights == True:
        column_name = list(data.columns)[-2]
    else:
        column_name = list(data.columns)[-1]

    hOmegas = data['hOmega'].unique()
    hOmegas.sort()

    #print('hOmegas = ', hOmegas)

    # t = np.empty(
    #     [len(hOmegas), 2])  # массив, который будет содержать Nmax, hOmega, для которых будут делаться предсказания
    # for i in range(len(hOmegas)):
    #     t[i][1] = hOmegas[i]  # типа hOmega # hw остается постоянным, меняется только Nmax

    # # print(t)

    # for N in range(maxN, n + 1, 2):  # цикл как бы по Nmax

    #     for i in range(len(hOmegas)):
    #         t[i][0] = N  # типа hOmega # hw остается постоянным, меняется только Nmax

    #     # print(t)

    #     for i in models:
    #         ts = scalerx.transform(t)
            
    #         ts = tf.convert_to_tensor(ts)

    #         ts_pred = i.predict(ts)

    #         predicted_value = scalery.inverse_transform(ts_pred)[:, 0]
    #         #r = scalery.inverse_transform(ts_pred)[:, 1]
    #         idx = models.index(i)
    #         for j in range(len(t)):  # разные hOmega
    #             preds.append([N, t[j][1], predicted_value[j], idx])
    
        
    arr_for_preds = []
    
    for N in range(maxN, n + 1, NMAX_PREDICTIONS_STEP):  # цикл как бы по Nmax
        for i in range(len(hOmegas)):
            arr_for_preds.append([N, hOmegas[i]])       
       
    
    arr_for_preds = np.array(arr_for_preds)    
    arr_for_preds = np.reshape(arr_for_preds, (-1,2))   
        
    for i in models:
        ts = scalerx.transform(arr_for_preds)                
        ts = tf.convert_to_tensor(ts)        
        ts_pred = i.predict(
            ts,
            batch_size=None,
            verbose=0,
            steps=None,
            callbacks=None,
            max_queue_size=10,
            workers=2,
            use_multiprocessing=False
            )

        predicted_value = scalery.inverse_transform(ts_pred)[:, 0]        
        idx = models.index(i)           
        
        
        
        ann_indeces = [] # номер сетки, столбец чтоб внести в датафрейм
        for i in range(len(arr_for_preds)):
            ann_indeces.append(idx)
         
        
        ann_indeces = np.array(ann_indeces).T
        #print(len(ann_indeces))
        
        predicted_value = np.array(predicted_value).T
        
        
        preds.extend(zip(arr_for_preds[:,0], arr_for_preds[:,1], predicted_value, ann_indeces))
    


    column_names = ["Nmax", "hOmega", column_name, "annid"]
    preds = pd.DataFrame(preds, columns=column_names)  # predictions

    return preds, loading_models_time

def cut_left_than_minE(df):
    # обрезание энергии левее минимума
    minN = df['Nmax'].min()
    # print(minN)
    maxN = df['Nmax'].max()
    # print(maxN)
    Nmaxes = df['Nmax'].unique()    
    for Nmax in Nmaxes:
        # print(Nmax)
        minE_idx = df[df['Nmax'] == Nmax]['Eabs'].idxmin()
        # print(minE_idx)
        corr_hO = df[df['Nmax'] == Nmax]['hOmega'][minE_idx]
        # print(corr_hO)
        df.drop(df[df['Nmax'] == Nmax][df.hOmega < corr_hO].index, inplace=True)

def assemble_loss_from_traisave_pieces(path, file_mask):
    # file_mask - маска файла: loss или val_loss, например
    # в trainsave при распараллеливании сохраняется loss в .csv
    # при этом сохраняется файл, в котором записан loss для сеток, которые обучал один процесс
    # эта функция склеивает эти файлы в один массив и сохраняет в то же место, файлы-кусочки удаляются
    # для того чтобы потом построить гистограммы
    loss_complete = pd.DataFrame()

    values = []
    indices = []

    for file in path.glob(file_mask + '*.csv'):
        print(f'assemble_loss_from_traisave_pieces: file = {file}')
        filename = file.stem
        begin_idx = int(re.search(r'\d+', filename).group(0))
        
        print(f'begin index = {begin_idx}')
        
        loss_partial = pd.read_csv(file, sep = '\t')
        
        #print(loss_partial)
        
        loss_partial.annid += begin_idx
        
        #print(loss_partial)
        
        values.extend(loss_partial[file_mask])
        indices.extend(loss_partial.annid)
        
        ## Try to delete the file ##
        try:
            os.remove(file)
        except OSError as e:  ## if failed, report it back to the user ##
            print ("Error: %s - %s." % (e.filename, e.strerror))
        
    loss_complete[file_mask]  = values
    loss_complete['annid'] = indices

    loss_complete = loss_complete.sort_values(by=['annid'])
    loss_complete.to_csv(os.path.join(path, (file_mask + '.csv')), sep='\t', index=False) 

def assemble_weights_from_traisave_pieces(path, file_mask):
    # file_mask - маска файла ("weights")
    # в trainsave при распараллеливании сохраняются веса в слоях в .csv
    # сохраняется файл, в котором записаны веса для сеток, которые обучал один процесс
    # эта функция склеивает эти файлы в один массив и сохраняет в то же место, файлы-кусочки удаляются
    # для того чтобы потом построить гистограммы
    data_complete = pd.DataFrame()

    weight_values = []
    layer_values  = []
    indices = []

    for file in path.glob(file_mask + '*.csv*'):

        filename_string = file.stem
        print(f'assemble_weights_from_traisave_pieces: filename_string = {filename_string}')

        filename = filename_string.split('weights')[-1] # split для того чтобы работало и для весов для разных эпох
        print(f'assemble_weights_from_traisave_pieces: filename = {filename}')
        
        begin_idx = int(re.search(r'\d+', filename).group(0))
        
        print(f'begin index = {begin_idx}')
        
        data_partial = pd.read_csv(file, sep = '\t')
        
        #print(data_partial)
        
        #data_partial.annid += begin_idx
        
        #print(data_partial)
        
        weight_values.extend(data_partial['weight'])
        layer_values.extend( data_partial['layer'])
        indices.extend(      data_partial['annid'])
        
        # Try to delete the file ##
        try:
            os.remove(file)
            print('file with partial data deleted')
        except OSError as e:  ## if failed, report it back to the user ##
            print ("Error: %s - %s." % (e.filename, e.strerror))
        
    data_complete['weight'] = weight_values
    data_complete['layer']  = layer_values
    data_complete['annid']  = indices

    data_complete = data_complete.sort_values(by=['annid', 'layer'])
    data_complete.to_csv(os.path.join(path, (file_mask + '.csv')), sep='\t', index=False) 

def assemble_mse_over_epochs_from_traisave_pieces(path, file_mask):
    '''
    функция для объединения значений mse в течение обучения
    file_mask - маска файла:
    в trainsave при распараллеливании сохраняется mse в .csv
    при этом сохраняется файл, в котором записан mse для сеток, которые обучал один процесс
    эта функция склеивает эти файлы в один массив и сохраняет в то же место, файлы-кусочки удаляются
    для того чтобы потом построить график усредненного по ансамблю mse в ходе обучения
    '''
    mse_complete = pd.DataFrame()

    epochs_list = []
    mse_list = []
    indices = []

    for file in path.glob(file_mask + '*.csv'):
        print(f'assemble_loss_from_traisave_pieces: file = {file}')
        filename = file.stem
        begin_idx = int(re.search(r'\d+', filename).group(0))
        
        print(f'begin index = {begin_idx}')
        
        loss_partial = pd.read_csv(file, sep = '\t')
        
        #print(loss_partial)
        
        loss_partial.annid += begin_idx
        
        #print(loss_partial)
        
        epochs_list.extend(loss_partial['epoch'])
        mse_list.extend(   loss_partial['mse'])
        indices.extend(    loss_partial['annid'])
        
        ## Try to delete the file ##
        try:
            os.remove(file)
        except OSError as e:  ## if failed, report it back to the user ##
            print ("Error: %s - %s." % (e.filename, e.strerror))

    mse_complete['epoch']  = epochs_list   
    mse_complete['mse']  = mse_list
    mse_complete['annid'] = indices

    mse_complete = mse_complete.sort_values(by=['annid', 'epoch'])
    mse_complete.to_csv(os.path.join(path, (file_mask + '.csv')), sep='\t', index=False, float_format = '%.5e')

def filter_mse_over_epochs(path, file_name: str):
    '''
    прореживание mse_over_epochs
    '''
    df = pd.read_csv(path / (file_name + '.csv'), sep='\t')
    filtered_df = df.iloc[::1000, :]    
    filtered_df.to_csv(path / (file_name + '_filtered.csv'), sep='\t', index=False, float_format = '%.5e')



#=====================================

if __name__ == '__main__':
    freeze_support()
    
    summary = open(path / 'train&predict_summary.txt', 'w', encoding = 'utf8') # текстовый файл с различными сведениями
    summary.write(f'BEGIN\t{t_begin}')
    summary.write(f'\npath\t{path}')
    summary.write(Notes)
    
    
    #чтение исходных данных
    if with_sample_weights == True:
        df_e = pd.read_excel(path_data_e, sheet_name = 'data', header = 0, usecols = columns_data_e + ',' + column_weight_e)
        df_r = pd.read_excel(path_data_r, sheet_name = 'data', header = 0, usecols = columns_data_r + ',' + column_weight_r)        
    else:
        df_e = pd.read_excel(path_data_e, sheet_name = 'data', header = 0, usecols = columns_data_e)
        df_r = pd.read_excel(path_data_r, sheet_name = 'data', header = 0, usecols = columns_data_r)  

    print('**** raw data ****')
    print(df_e)
    print('********* \n **** raw data ****')
    print(df_r)
    print('********* \n')
    
    
    # СДЕЛАТЬ НОРМАЛЬНО
    #============================================
    # if cut_horizontal_EandR == True:
        
    #     df_e.drop(df[df.Eabs > E_horizontal].index, inplace=True)
    # #============================================
        
     
        
    # ограничиваем исходные данные по hOmega (слева и справа) и по макс. Nmax (сверху и снизу)
    # ============================================
    # для энергии

    # выделение данных для подсчета метрики
    df_for_metric_e = df_e[df_e['Nmax'] == Nmax_for_metric_e]
    # и отбрасывание их из исходного датасета
    df_e.drop(df_e[df_e.Nmax == Nmax_for_metric_e].index, inplace=True)



    df_e.drop(df_e[df_e.Nmax > maxN_e].index,         inplace=True)  # убираем строки, где Nmax > maxN
    df_e.drop(df_e[df_e.Nmax < minN_e].index,         inplace=True)
    df_e.drop(df_e[df_e.hOmega < hwcutoff_l_e].index, inplace=True)
    df_e.drop(df_e[df_e.hOmega > hwcutoff_r_e].index, inplace=True)   
               
    if cut_on_min == True: # обрезание энергии левее минимума
        cut_left_than_minE(df_e)
                      
    if horizontal_cut == True: # обрезание энергии, лежащей выше некоторой E_horizontal
        df_e.drop(df_e[df_e.Eabs > E_horizontal].index, inplace=True) 
        
        
    hwcutoff_l_e = df_e['hOmega'].min()  # переопределяем левую границу hOmega
    
    # определяем вариационный минимум здесь, т.е. в ограниченных данных, но без отбрасывания данных для метрики
    variational_min = df_e['Eabs'].min()  # переопределяем вариационный минимум
    print(variational_min)
    summary.write(f'\n{variational_min}\t вариационный минимум в выбранных даннных')
    
    
       
    
    
    # для радиуса
    df_for_metric_r = df_r[df_r['Nmax'] == Nmax_for_metric_r] # для метрики
    df_r.drop(df_r[df_r.Nmax == Nmax_for_metric_r].index, inplace=True)

    df_r.drop(df_r[df_r.Nmax > maxN_r].index,         inplace=True)  
    df_r.drop(df_r[df_r.Nmax < minN_r].index,         inplace=True)
    df_r.drop(df_r[df_r.hOmega < hwcutoff_l_r].index, inplace=True)
    df_r.drop(df_r[df_r.hOmega > hwcutoff_r_r].index, inplace=True)
    
    # веса семплов при обучении
    if with_sample_weights == True:        
        sample_weights_e = df_e['Weight'].to_numpy()
        sample_weights_r = df_r['Weight'].to_numpy()
    else:
        sample_weights_e = None
        sample_weights_r = None
    
    # не используется (старый артефакт)
    neurons_e = len(df_e)
    neurons_r = len(df_r)
    
    #==========================================================================
    if scaler_in_min_e == None:
        scaler_in_min_e = df_e['Nmax'].min() 
    if scaler_in_max_e == None:
        scaler_in_max_e = df_e['Nmax'].max()
    
    if scaler_in_min_r == None:
        scaler_in_min_r = df_r['Nmax'].min() 
    if scaler_in_max_r == None:
        scaler_in_max_r = df_r['Nmax'].max()
    #==========================================================================
    
    print(df_e)
    print(df_r)
    
        
    # df_e.plot(x="hOmega", y="Eabs", kind='scatter', color='tomato', s=240)
    # df_r.plot(x="hOmega", y="RMS",  kind='scatter', color='gold',   s=240)
    # plt.show()
    
    # используемые данные 
    # более темным цветом - для метрики (тестовые данные)
    # plt.scatter(df_e['hOmega'], df_e['Eabs'], color = 'tomato', s=20)
    # plt.scatter(df_for_metric_e['hOmega'], df_for_metric_e['Eabs'], color = 'firebrick', s=20)
    # plt.ylabel('$Eabs$')
    # plt.xlabel('$\hbar \Omega$')
    # plt.savefig(pics_path_e / ('Используемые данные.' + plot_format), format = plot_format)
    # #plt.show()
    # plt.close()
    
    
    # plt.scatter(df_r['hOmega'], df_r['RMS'], color = 'gold', s=20)
    # plt.scatter(df_for_metric_r['hOmega'], df_for_metric_r['RMS'], color = 'goldenrod', s=20)
    # plt.ylabel('$RMS$')
    # plt.xlabel('$\hbar \Omega$')
    # plt.savefig(pics_path_r / ('Используемые данные.' + plot_format), format = plot_format)
    # #plt.show()
    # plt.close()
    
    
    for i in range(numofnns_e):
        save_path = path_e / str(i)
        save_path.mkdir(parents=False, exist_ok=True) # создание папок для сохранения моделей 
        
    for i in range(numofnns_r):
        save_path = path_r / str(i)
        save_path.mkdir(parents=False, exist_ok=True) # создание папок для сохранения моделей
    
    
    # запись гиперпараметров и прочего в файл summary
    #---------------------------------------------------------------------
    summary.write(Notes)        
    summary.write(f'\nvariational_min = {variational_min}\tвариационный минимум энергии в используемых данных')
    
    summary.write('\n\nГиперпарамеры')
    summary.write('\n----------------------------------------------------')
    
    summary.write(f'\nextra fitting = {EXTRA_FITTING}, дополнительное обучение в течение нескольких эпох с некоторым batch size и learn rate = LR_e(r)')
    summary.write(f'\nextra epochs = {EPOCHS_EXTRA}')
    summary.write(f'\nbatch size = {BS_EXTRA}')
    summary.write(f'\nlearning rate extra coef. = {LR_EXTRA_COEF}, lr_extra  = lr * lr_extra_coef')
    
    summary.write('\nCyclicalLearningRate\n')
    summary.write(f'\nCLR_NUM_CYCLES = {CLR_NUM_CYCLES}\t# число циклов в cyclical_learning_rate')
    summary.write(f'\nCLR_COEF_MAXLR = {CLR_COEF_MAXLR}\tLR in [base LR, max LR], max LR = CLR_COEF_MAXLR * base LR')
    
    
    
    summary.write('\n\nEnergy\n')    
    
    summary.write(f'\ncut_on_min = {cut_on_min}\tобрезание энергии левее минимума')
    summary.write(f'\nhorizontal_cut = {horizontal_cut}\tобрезание энергии, которая лежит выше некоторой E_horizontal')
    summary.write(f'\nE_horizontal = {E_horizontal}\tгоризонтальная энергия обрезания')  
    summary.write(f'\nAFC_e = {AFC_e}\tactivation function coefficient')
    summary.write(f'\nAFC2_e = {AFC2_e}')
    summary.write(f'\nAFB_e = {AFB_e}\tactivation function bias')
    summary.write(f'\nNE_e = {NE_e}\tnum of epochs')
    summary.write(f'\nBS_e = {BS_e}\tbatch size')
    summary.write(f'\nLR_e = {LR_e}\tlearning rate')
    summary.write(f'\nLRD_e = {LRD_e}\tlearning rate decay')
    summary.write(f'\nnumofnns_e = {numofnns_e}\tчисло нейросетей, с которого стартуем')
    summary.write(f'\nn_e = {n_e}\tNmax, до которого постепенно будем предсказывать: N = maxN, maxN+2, .. ,n')
    
    
    summary.write('\n\nпараметры скейлера (предполагается для minmax scaler):\n')
    summary.write(f'\nscaler_in_min_e = {scaler_in_min_e}\tминимум  для входа')
    summary.write(f'\nscaler_in_max_e = {scaler_in_max_e}\tмаксимум для входа')
    summary.write(f'\nscaler_out_min_e = {scaler_out_min_e}\tминимум  для выхода')
    summary.write(f'\nscaler_out_max_e = {scaler_out_max_e}\tмаксимум для выхода\n')
    summary.write(f'\nminN_e = {minN_e}\tминимальное учитываемое модельное пространство')
    summary.write(f'\nmaxN_e = {maxN_e}\tмаксимальное учитываемое модельное пространство')
    summary.write(f'\nhwcutoff_l_e = {hwcutoff_l_e}\tобрезание слева по hOmega')
    summary.write(f'\nhwcutoff_r_e = {hwcutoff_r_e}\tобрезание справа по hOmega')
   
    
    summary.write('\n\nRadius\n')
    
    summary.write(f'\nAFC_r = {AFC_r}\tactivation function coefficient')
    summary.write(f'\nAFC2_r = {AFC2_r}')
    summary.write(f'\nAFB_r = {AFB_r}\tactivation function bias')
    summary.write(f'\nNE_r = {NE_r}\tnum of epochs')
    summary.write(f'\nBS_r = {BS_r}\tbatch size')
    summary.write(f'\nLR_r = {LR_r}\tlearning rate')
    summary.write(f'\nLRD_r = {LRD_r}\tlearning rate decay')
    summary.write(f'\nnumofnns_r = {numofnns_r}\tчисло нейросетей, с которого стартуем')
    summary.write(f'\nn_r = {n_r}\tNmax, до которого постепенно будем предсказывать: N = maxN, maxN+2, .. ,n')
    summary.write('\n\nпараметры скейлера (предполагается для minmax scaler):\n')
    summary.write(f'\nscaler_in_min_r = {scaler_in_min_r}\tминимум  для входа')
    summary.write(f'\nscaler_in_max_r = {scaler_in_max_r}\tмаксимум для входа')
    summary.write(f'\nscaler_out_min_r = {scaler_out_min_r}\tминимум  для выхода')
    summary.write(f'\nscaler_out_max_r = {scaler_out_max_r}\tмаксимум для выхода\n')
    summary.write(f'\nminN_r = {minN_r}\tминимальное учитываемое модельное пространство')
    summary.write(f'\nmaxN_r = {maxN_r}\tмаксимальное учитываемое модельное пространство')
    summary.write(f'\nhwcutoff_l_r = {hwcutoff_l_r}\tобрезание слева по hOmega')
    summary.write(f'\nhwcutoff_r_r = {hwcutoff_r_r}\tобрезание справа по hOmega')
    
    summary.write('\n----------------------------------------------------')
#--------------------------------------------------------------------



# ограничили данные


# ================================================
if __name__ == '__main__':
    freeze_support()
    # подготовка данных для энергии
    # Changing pandas dataframe to numpy array
    x_e = df_e[['Nmax', 'hOmega']].to_numpy()
    y_e = df_e[['Eabs']].to_numpy()
    # print(x_e)
    # print(y_e)
    
    scalerx_e = MinMaxScaler(feature_range=(scaler_in_min_e,  scaler_in_max_e))   # скейлер для входных данных # -4 4
    scalery_e = MinMaxScaler(feature_range=(scaler_out_min_e, scaler_out_max_e))  # скейлер для выходных данных
    
    scalerx_e.fit(x_e)
    scalery_e.fit(y_e)
    
    x_e = scalerx_e.transform(x_e)
    y_e = scalery_e.transform(y_e)
    
    # подготовка данных для радиуса
    x_r = df_r[['Nmax', 'hOmega']].to_numpy()
    y_r = df_r[['RMS']].to_numpy()
    # print(x_r)
    # print(y_r)
    
    scalerx_r = MinMaxScaler(feature_range=(scaler_in_min_r,  scaler_in_max_r))  # скейлер для входных данных # -4 4
    scalery_r = MinMaxScaler(feature_range=(scaler_out_min_r, scaler_out_max_r))  # скейлер для выходных данных
    
    scalerx_r.fit(x_r)
    scalery_r.fit(y_r)
    
    x_r = scalerx_r.transform(x_r)
    y_r = scalery_r.transform(y_r)    
# =============================================================================

#==============================================================================
# определение активирующих функций
# проще способа не нашел
#get_custom_objects().update({'custom_activation': Activation(custom_activation)})

def custom_activation_e(x):    
    return (keras.backend.sigmoid(AFC_e * x) + AFB_e) * AFC2_e #0.5 #0.15

def custom_activation_r(x):    
    return (keras.backend.sigmoid(AFC_r * x) + AFB_r) * AFC2_r #0.5 #0.15



#==============================================================================
def trainsave(path, numofnns, begin, x, y, splitting, learn_rate, lr_decay, num_epochs, batch_s, activ_func, neurons, 
              with_return = False, scale_ymax = None, scale_ymin = None, scale_max = None, scale_min = None, sample_weights_fit = None):
    
    #import keras
    #import tensorflow as tf
    #print(with_return)
    # splitting - если true, то есть разбивка на тренировочный и валидационные наборы +
    # + запись в файл ошибки на валидационном наборе
    # x, y - признаки и ответы
    # path - папка для сохранения моделей и другого
    # begin - начальный номер для моделей, т.е. модели будут иметь номера от begin
    # path - путь для сохранения моделей
    # neurons - количество нейронов в пером слое
    # scale_ для обратной конвертации значений loss'a (формулка написана только для предоположения MinMaxScaler)
    # with_sample_weights - веса для данных
    
    #numofnns = numofnns_end - numofnns_begin
    
    # models = [None] * numofnns
    # histories = [None] * numofnns
    #print('trainsave: path =', path)
    #print('trainsave: splitting = ', splitting)
    #print('trainsave: numofnns = ',  numofnns)
    print('trainsave: begin = ',  begin)      

    weights_df_dict = {a: pd.DataFrame(columns = ['weight', 'layer', 'annid']) for a in EPOCHS_FOR_SAMPLING}
    # словарь словарей со списками, которые будут заполняться данными. Ключом так же является номер эпохи
    weights_data_dict = {a: {} for a in EPOCHS_FOR_SAMPLING}
    for i in EPOCHS_FOR_SAMPLING:
        weights_data_dict[i] = {'weights_col_all': [], 'layer_col_all': [], 'annid_col_all': []}

    class write_weights_callback(tf.keras.callbacks.Callback): # callback для записи весов         
        def on_epoch_begin(self, epoch, logs=None):               
            if epoch in EPOCHS_FOR_SAMPLING:
                #print(f'write_weights_callback! epoch {epoch}')
                
                model_id = self.model.name
                #print(f'write_weights_callback = {model_id}')
                    
                # weights_col_all = [] # массивы-колонки для всех сеток
                # layer_col_all   = []
                # annid_col_all   = []
                #for i in range(numofnns):                    
                for layer in range(len(self.model.layers)):                
                    weights_col = np.ravel(self.model.layers[layer].get_weights()[0]) # ravel дает одномерный массив                
                    number_of_weights = len(weights_col) # количество весов в слое
                    
                    # заполнение столбцов с номером слоя и номером сетки
                    layer_col = np.empty(number_of_weights)                
                    layer_col.fill(layer) # столбец заполнен номером слоя                

                    annid_col = np.empty(number_of_weights)                
                    annid_col.fill(model_id) # столбец заполнен номером сетки

                    #print(f'len(weights_col) = {len(weights_col)}')
                    #print(f'len(layer_col) = {  len(layer_col)}')
                    #print(f'len(annid_col) = {  len(annid_col)}')
                    
                    # weights_col_all.extend(weights_col)
                    # layer_col_all.extend(layer_col)
                    # annid_col_all.extend(annid_col)
                    weights_data_dict[epoch]['weights_col_all'].extend(weights_col)
                    weights_data_dict[epoch][  'layer_col_all'].extend(layer_col)
                    weights_data_dict[epoch][  'annid_col_all'].extend(annid_col)

                    #print(f"len(weights_col_all) = {len(weights_data_dict[epoch]['weights_col_all'])}")
                    #print(f"len(layer_col_all) = {  len(weights_data_dict[epoch]['layer_col_all'])}")
                    #print(f"len(annid_col_all) = {  len(weights_data_dict[epoch]['annid_col_all'])}")

                    #time.sleep(3) 
    
    if batch_s == None:
        batch_s = len(x)
        
    models = []
    histories = []
    additional_histories = []
    
    val_loss = []  # ошибка на валидационном наборе в последнюю эпоху, если splitting = true
    loss = []
 
         
    
    # словарь датафреймов для сохранения весов всех моделей: для каждой эпохи свой датафрейм. Ключом является номер эпохи
    weights_df_dict = {a: pd.DataFrame(columns = ['weight', 'layer', 'annid']) for a in EPOCHS_FOR_SAMPLING}
    # словарь словарей со списками, которые будут заполняться данными. Ключом так же является номер эпохи
    weights_data_dict = {a: {} for a in EPOCHS_FOR_SAMPLING}
    for i in EPOCHS_FOR_SAMPLING:
        weights_data_dict[i] = {'weights_col_all': [], 'layer_col_all': [], 'annid_col_all': []}

    # обучаем некоторое количество - numofnns - нейросеток на начальных данных
    for i in range(numofnns):       
        print(f'\ntraining network № {begin + i}...')
                
        models.append(Sequential(name = str(begin + i)))
        
        #models[i] = Sequential()

        # models[i].add(Dense(30, input_dim=2, activation="linear", name="1st"))
        # models[i].add(Dense(30, activation="sigmoid", name="2nd"))
        # models[i].add(Dense(30, activation="sigmoid", name="3rd"))
        # models[i].add(Dense(30, activation="sigmoid", name="4th"))
        # models[i].add(Dense(2,  activation="linear", name="output"))
        
        
        
        
        initializer = tf.keras.initializers.GlorotUniform(secrets.randbelow(100000)) # инициализация весов
        #initializer = tf.keras.initializers.GlorotNormal()
        #initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None)
        #initializer = tf.keras.initializers.RandomUniform(minval=-10.0, maxval=10.0)
 
        #models[i].add(Dense(150, input_dim=2,   activation = "linear", name="1st", kernel_initializer=initializer, use_bias=bias_usage))      
        #models[i].add(tf.keras.layers.Rescaling(0.1, offsactet = 0.0))
        #models[i].add(Dense(50,               activation = activ_func, name="2nd", kernel_initializer=initializer, use_bias=bias_usage))  
        #models[i].add(tf.keras.layers.Rescaling(0.5, offset=0el.0))
        #models[i].add(Dense(50,                activation = activ_func, name="3rd", kernel_initializer=initializer, use_bias=bias_usage)) 
        #models[i].add(tf.keras.layers.Rescaling(0.1, offset=0.0))
        #models[i].add(Dense(100,                activation = activ_func, name="4th", kernel_initializer=initializer, use_bias=bias_usage)) 
        #models[i].add(tf.keras.layers.Rescaling(0.1, offset=0.0))
        #models[i].add(Dense(10,               activation = activ_func, name="5th", kernel_initializer=initializer, use_bias=bias_usage)) 
        #models[i].add(Dense(1,                 activation = "linear",   name="output"))
        
        
        models[i].add(Dense(10, input_dim=2,   activation = "linear",   name="1st", kernel_initializer=initializer, use_bias=bias_usage))        
        #models[i].add(tf.keras.layers.Rescaling(0.1, offsactet = 0.0))
        #models[i].add(tfa.layers.WeightNormalization(Dense(150, input_dim=2,  activation = "linear",   name="1st", kernel_initializer=initializer, use_bias=bias_usage )))
        models[i].add(Dense(10,                activation = activ_func, name="2nd", kernel_initializer=initializer, use_bias=bias_usage))        
        #models[i].add(tf.keras.layers.Rescaling(10.0, offset=0.0))
        models[i].add(Dense(10,                activation = activ_func, name="3rd", kernel_initializer=initializer, use_bias=bias_usage))#, kernel_constraint = tf.keras.constraints.non_neg() ))
        #models[i].add(tf.keras.layers.Rescaling(10.0, offset=0.0))        
        #models[i].add(Dense(10,                activation = activ_func, name="4th", kernel_initializer=initializer, use_bias=bias_usage )) 
        #models[i].add(tf.keras.layers.Rescaling(10.0, offset=0.0))
        #models[i].add(Dense(10,                activation = activ_func, name="5th", kernel_initializer=initializer, use_bias=bias_usage )) 
        #models[i].add(tf.keras.layers.Rescaling(10.0, offset=0.0))
        #models[i].add(Dense(50,                activation = activ_func, name="6th", kernel_initializer=initializer, use_bias=bias_usage )) 
        #models[i].add(tf.keras.layers.Rescaling(10.0, offset=0.0))
        #models[i].add(Dense(50,                activation = activ_func, name="7th", kernel_initializer=initializer, use_bias=bias_usage ))
        #models[i].add(tf.keras.layers.Rescaling(10.0, offset=0.0))
        #models[i].add(Dense(50,                activation = activ_func, name="8th", kernel_initializer=initializer, use_bias=bias_usage ))
        models[i].add(Dense(1,                 activation = "linear",   name="output"))

        #models[i].summary()
        
        #lr_schedule = learn_rate0004x40
        # initial_learning_rate = learn_rate*10
        # lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        #     initial_learning_rate,
        #     decay_steps = NE,
        #     decay_rate  = lr_decay,
        #     staircase   = False)

        # initial_learning_rate = learn_rate
        # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        #     initial_learning_rate,
        #     decay_steps = 100,
        #     decay_rate  = lr_decay,
        #     staircase   = False)
        
        # initial_learning_rate = 100*learn_rate
        # decay_steps = 10.0
        # decay_rate = 0.5
        # lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        #   initial_learning_rate, decay_steps, decay_rate)
        
        INIT_LR = learn_rate
        MAX_LR  = CLR_COEF_MAXLR * learn_rate
        clr_num_cycles = CLR_NUM_CYCLES
              
        steps_per_epoch = len(x) // batch_s
        clr_step_size = (num_epochs * steps_per_epoch) // (2 * clr_num_cycles) # 2x пушо step_size это половина цикла
        
        #print(clr_step_size)        
        # print(batch_s)
        # print(len(x))
        # print(steps_per_epoch)
              
        clr = tfa.optimizers.CyclicalLearningRate(initial_learning_rate = INIT_LR,
        maximal_learning_rate = MAX_LR,
        scale_fn = scale_fn,
        #scale_fn = lambda x: 1/(2.**(x-1)),
        #step_size = int(num_epochs / 5.0)
        step_size = clr_step_size #4 * steps_per_epoch #10000 #400 #100 #100 ok
        )
        
        
        #построение зависимости скорости обучения от шага        
        step = np.arange(0, num_epochs * steps_per_epoch)        
        lr = clr(step)    
        sc_fu = INIT_LR + scale_fn(step/(2*clr_step_size)) * MAX_LR
        minimal_scfu = min(sc_fu)
        plt.title(f'min(scale function) = {minimal_scfu:.2e}')
        plt.plot(step, lr)
        plt.plot(step, sc_fu)
        plt.ylim(top=MAX_LR)
        plt.ylim(bottom = 0)
        plt.xlabel("Steps")
        plt.ylabel("Learning Rate")
        #plt.show()
        plt.savefig(path / ('learning_rate_schedule.' + plot_format), format = plot_format)
        plt.close()
        
        
        callbacks = []
        callbacks.append(new_callback())

        if WRITE_WEIGHTS == True:
            callbacks.append(write_weights_callback())
        
        callbacks.append(tf.keras.callbacks.EarlyStopping(
        monitor              = 'loss',
        min_delta            = 0,
        patience             = EARLY_STOPPING_2_PATIENCE,
        verbose              = 1,
        mode                 = 'auto',
        baseline             = None,
        restore_best_weights = True
        ))
        
        
        # разные оптимизаторы для разных слоев
        # #=====================================
        # optimizers = [
        # tf.keras.optimizers.Adam(learning_rate = learn_rate),
        # tfa.optimizers.NovoGrad(learning_rate = clr)
        # ]
        
        # optimizers_and_layers = [(optimizers[0], models[i].layers[0]), (optimizers[1], models[i].layers[1:])]
        # optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)
        # #=====================================
        
        
        models[i].compile(
            loss = tf.keras.losses.MeanSquaredError(), \
            #loss = tf.keras.losses.poisson, \
            #loss = tf.keras.losses.MeanAbsoluteError(), \
            # loss = custom_loss, \
            #optimizer = optimizer
            optimizer = tf.keras.optimizers.Adam(learning_rate = clr), \
            #optimizer = tfa.optimizers.NovoGrad(clr),
            #optimizer = tf.keras.optimizers.SGD(learning_rate = lr_schedule, momentum=0.0, nesterov=False, name="SGD"),\
            #optimizer = tf.keras.optimizers.SGD(learning_rate = clr),
            #optimizer = tf.keras.optimizers.Adadelta(learning_rate = clr, rho=0.95, epsilon=1e-07),
            metrics    = [tf.keras.metrics.MeanSquaredError()],         
            #metrics = [tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanSquaredError()] #, lr_metric
            )
        
        
            
        # splitting data into training and testing data, training model
        if splitting == True:
                        
            if sample_weights_fit is not None:
                x_train, x_test, y_train, y_test, sample_weights_train, sample_weights_test = train_test_split(x, y, sample_weights_fit, test_size=0.3, shuffle = True)
            else:
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle = True)
                        
            histories.append(models[i].fit(
                x_train, y_train, \
                validation_data=(x_test, y_test), \
                epochs = num_epochs, batch_size = batch_s, \
                shuffle = SHUFFLE, \
                sample_weight = sample_weights_train, \
                #use_multiprocessing = True, \
                #workers = 4, \
                # callbacks=[decay], \
                callbacks = callbacks,
                initial_epoch = 0,
                verbose=VERBOSITY))
            
            if EXTRA_FITTING == True:
                
                print(f'extra fitting for № {begin + i}...')  
                
                initial_learning_rate = LR_EXTRA_COEF * learn_rate
                lr_schedule_extra = tf.keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate,
                    decay_steps = len(x) // BS_EXTRA,
                    decay_rate  = lr_decay,
                    staircase   = False)
                
                
                models[i].compile(
                    loss = tf.keras.losses.MeanSquaredError(),
                    #loss      = tf.keras.losses.MeanAbsoluteError(),
                    optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule_extra),
                    metrics   = [tf.keras.metrics.MeanSquaredError()])
                callbacks_extra = []

                if WRITE_WEIGHTS == True:
                    callbacks_extra.append(write_weights_callback())
                
                callbacks_extra.append(tf.keras.callbacks.EarlyStopping(
                monitor              = 'loss',
                min_delta            = 0,
                patience             = EARLY_STOPPING_2_PATIENCE_EXTRA,
                verbose              = 1,
                mode                 = 'auto',
                baseline             = None,
                restore_best_weights = True
                ))
                
                additional_histories.append(models[i].fit(
                    x_train, y_train, \
                    validation_data=(x_test, y_test), \
                    epochs = EPOCHS_EXTRA, batch_size = BS_EXTRA, \
                    shuffle = SHUFFLE_EXTRA, \
                    sample_weight = sample_weights_train,
                    #use_multiprocessing = True, \
                    #workers = 4, \
                    # callbacks=[decay], \
                    callbacks = callbacks_extra,
                    initial_epoch = 0,
                    verbose = VERBOSITY))
                    
                val_loss.append([additional_histories[i].history['val_mean_squared_error'][-1], i]) # в последнюю эпоху
                loss.append(    [additional_histories[i].history[    'mean_squared_error'][-1], i]) # в последнюю эпоху
                
                
                
                # summarize ADDITIONAL history for loss
                #------------------------------------------------------
                if training_threading == False:
                    plt.plot(additional_histories[i].history['loss'])
                    plt.plot(additional_histories[i].history['val_loss'])
                    #plt.xlim(left=500) 
                    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
                    #plt.ylim(top=1e-6)
                    #plt.ylim(bottom = 0.0)         
                
                
                end_val_loss = additional_histories[i].history['val_mean_squared_error'][-1]
                if scale_ymax is not None and scale_ymin is not None and scale_max is not None and scale_min is not None:
                    end_val_loss_sqrt = math.sqrt(end_val_loss) 
                    #print(f'sqrt(min. val. loss) = {end_val_loss_sqrt}')
                    appr_err = end_val_loss_sqrt  * (scale_ymax - scale_ymin) / (scale_max - scale_min) # примерная ошибка
                    if training_threading == False:
                        plt.title(f'model {begin + i} loss [extra], end val. loss = {end_val_loss:.2e}, appr. err. = {appr_err:.2e}')
                else:
                    if training_threading == False:
                        plt.title(f'model {begin + i} loss [extra], end val. loss = {end_val_loss:.2e}')
                        
                if training_threading == False:
                    plt.ylabel('loss')
                    plt.xlabel('extra epoch')
                    plt.legend(['train', 'test'], loc='upper left', fontsize=40)
                    plt.show()
                
            
            else:
                val_loss.append([histories[i].history['val_mean_squared_error'][-1], i]) # в последнюю эпоху
                loss.append(    [histories[i].history[    'mean_squared_error'][-1], i]) # в последнюю эпоху
                
            #------------------------------------------------------
            
            
            # summarize history for loss
            #------------------------------------------------------
            if training_threading == False:
                plt.plot(histories[i].history['mean_squared_error'])
                plt.plot(histories[i].history['val_mean_squared_error'])
                #plt.xlim(left=500)
                plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
                #plt.ylim(top=1e-6)
                plt.ylim(bottom = 0.0)         
            
            
            end_val_loss = histories[i].history['val_mean_squared_error'][-1]
            if scale_ymax is not None and scale_ymin is not None and scale_max is not None and scale_min is not None:
                end_val_loss_sqrt = math.sqrt(end_val_loss)
                #print(f'scale_ymax = {scale_ymax}, scale_ymin = {scale_ymin} \nscale_max = {scale_max}, scale_min = {scale_min}')
                appr_err = end_val_loss_sqrt  * (scale_ymax - scale_ymin) / (scale_max - scale_min) # примерная ошибка
                if training_threading == False:
                    plt.title(f'model {begin + i} loss, end val. loss = {end_val_loss:.2e}, appr. err. = {appr_err:.2e}')
            else:
                if training_threading == False:
                    plt.title(f'model {begin + i} loss, end val. loss = {end_val_loss:.2e}')
                        
                      
            if training_threading == False:
                plt.ylabel('loss')
                plt.xlabel('epoch')
                plt.legend(['train', 'test'], loc='upper left', fontsize=40)
                plt.show()            
            #------------------------------------------------------       
              
            # saving models            
            save_path = path / str(begin + i)
            print('trainsave: save path = ', save_path)
            
            try:
                #save_path.mkdir(parents=True, exist_ok=True) # создание папки для сохранения модели            
                models[i].save(save_path)
                print(f'\nnetwork № {begin + i} saved...', flush=True)
            except Exception as E:
                #os.system("pause")
                #os._exit(0)
                try:
                    time.sleep(1)
                    models[i].save(save_path)
                    print(f'\nnetwork № {begin + i} saved: 2nd attempt...', flush=True)
                except Exception as Ex:
                    print('SAVING ERROR after 2nd attempt')
                    print(Ex)
                    
                print('SAVING ERROR')
                print(E)


        else: # если splitting != true
                 
            x_train = x
            y_train = y

            sample_weights_train = sample_weights_fit
            
            # model_ = models[i].fit(x=x_train, y=y_train, epochs = num_epochs, batch_size = batch_s, shuffle=True, use_multiprocessing = False, verbose=VERBOSITY)
            
            # histories.append(model_)
            
            histories.append(models[i].fit(
                x_train, y_train,
                # validation_data = (x_test,y_test), \
                epochs = num_epochs, batch_size = batch_s,
                shuffle = SHUFFLE,
                sample_weight = sample_weights_train,
                use_multiprocessing = False,
                #workers = 4, \
                callbacks = callbacks,
                verbose = VERBOSITY))
            #print(histories[i].history.keys())
            
            
            if EXTRA_FITTING == True:           
                print(f'extra fitting for № {begin + i}...')           
                
                initial_learning_rate = LR_EXTRA_COEF * learn_rate
                lr_schedule_extra = tf.keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate,
                    decay_steps = len(x) // BS_EXTRA,
                    decay_rate  = lr_decay,
                    staircase   = False)
                
                
                models[i].compile(
                    loss = tf.keras.losses.MeanSquaredError(),
                    optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule_extra),
                    metrics    = [tf.keras.metrics.MeanSquaredError()])   
                callbacks_extra = []
                
                callbacks_extra.append(tf.keras.callbacks.EarlyStopping(
                monitor              = 'loss',
                min_delta            = 0,
                patience             = 400,
                verbose              = 1,
                mode                 = 'auto',
                baseline             = None,
                restore_best_weights = True
                ))
                               
                additional_histories.append(models[i].fit(
                    x_train, y_train,            
                    epochs = EPOCHS_EXTRA, batch_size = BS_EXTRA, \
                    shuffle = SHUFFLE_EXTRA, \
                    sample_weight = sample_weights_train, \
                    #use_multiprocessing = True, \
                    #workers = 4, \
                    # callbacks=[decay], \
                    callbacks = callbacks_extra,
                    initial_epoch = 0,
                    verbose=VERBOSITY))

                
                    
                loss.append([additional_histories[i].history['mean_squared_error'][-1], i]) # в последнюю эпоху              
                              
                
            else:         
                loss.append([histories[i].history['mean_squared_error'][-1], i]) # в последнюю эпоху
                
                            
            #------------------------------------------------------           
            # saving models            
            save_path = path / str(begin + i)            
            print('trainsave: save path = ', save_path)
            try:
                #save_path.mkdir(parents=True, exist_ok=True) # создание папки для сохранения модели            
                models[i].save(save_path)
                print(f'\nnetwork № {begin + i} saved...', flush=True)
            except Exception as E:
                #os.system("pause")
                #os._exit(0)
                try:
                    time.sleep(1)
                    models[i].save(save_path)
                    print(f'\nnetwork № {begin + i} saved: 2nd attempt...', flush=True)
                except Exception as Ex:
                    print('SAVING ERROR after 2nd attempt')
                    print(Ex)
                
                print('SAVING ERROR')
                print(E)        
                

    # сохранение loss на последней эпохе в файл
    if splitting == True:
        val_loss = pd.DataFrame(val_loss, columns=['val_loss', 'annid'])
        val_loss.to_csv(os.path.join(path, 'val_loss' + str(begin)+ '.csv'), sep='\t', index=False)
        
        loss = pd.DataFrame(loss, columns=['loss', 'annid'])
        loss.to_csv(os.path.join(path, 'loss' + str(begin)+ '.csv'), sep='\t', index=False)
    else:
        loss = pd.DataFrame(loss, columns=['loss', 'annid'])
        loss.to_csv(os.path.join(path, 'loss' + str(begin)+ '.csv'), sep='\t', index=False)

    #---------------------------------------------------------------------------------------------
    # сохранение mse по эпохам в файл
    mse_over_epochs_list = []
    epochs_list = []
    annids_list = []
    
    for annid in range(numofnns):
        print(f'annid={annid}')    
        #print(f'mse={mse_over_epochs_list}')
        mse_over_epochs_list.extend(histories[annid].history['mean_squared_error'])
        epochs_list.extend(list(range(num_epochs)))
        annids_list.extend([annid]*(num_epochs)) 

    hist_df = pd.DataFrame()
    hist_df['epoch'] = epochs_list   
    hist_df['mse'  ] = mse_over_epochs_list  
    hist_df['annid'] = annids_list    
    hist_df.to_csv(os.path.join(path, 'mse_over_epochs' + str(begin)+ '.csv'), sep='\t', index=False, float_format = '%.5e')
    del epochs_list
    del annids_list
    del mse_over_epochs_list
    #---------------------------------------------------------------------------------------------

    #---------------------------------------------------------------------------------------------
    # сохранение mse по дополнительным эпохам в файл
    if EXTRA_FITTING == True:
        mse_over_epochs_extra_list = []
        epochs_list = []
        annids_list = []
        
        for annid in range(numofnns):                        
            mse_over_epochs_extra_list.extend(additional_histories[annid].history['mean_squared_error'])
            epochs_list.extend(list(range(num_epochs)))
            annids_list.extend([annid]*num_epochs) 

        hist_df = pd.DataFrame()
        hist_df['extra epoch'] = epochs_list   
        hist_df['mse'  ] = mse_over_epochs_extra_list  
        hist_df['annid'] = annids_list    
        hist_df.to_csv(os.path.join(path, 'mse_over_extra_epochs' + str(begin)+ '.csv'), sep='\t', index=False, float_format = '%.5e')
        del epochs_list
        del annids_list
        del mse_over_epochs_extra_list
    
    #---------------------------------------------------------------------------------------------

    
    # запись весов между слоями в файл        
    if WRITE_WEIGHTS == True:        
        weights_col_all = [] # массивы-колонки для всех сеток
        layer_col_all   = []
        annid_col_all   = []
        for i in range(numofnns):                    
            for layer in range(len(models[i].layers)):                
                weights_col = np.ravel(models[i].layers[layer].get_weights()[0]) # ravel дает одномерный массив                
                number_of_weights = len(weights_col) # количество весов в слое
                
                # заполнение столбцов с номером слоя и номером сетки
                layer_col = np.empty(number_of_weights)                
                layer_col.fill(layer) # столбец заполнен номером слоя                

                annid_col = np.empty(number_of_weights)                
                annid_col.fill(begin + i) # столбец заполнен номером сетки
                
                weights_col_all.extend(weights_col)
                layer_col_all.extend(layer_col)
                annid_col_all.extend(annid_col)
        
        weights_df = pd.DataFrame(columns = ['weight', 'layer', 'annid'])
        weights_df['weight'] = weights_col_all
        weights_df['layer']  = layer_col_all
        weights_df['annid']  = annid_col_all

        # конвертация из вещественных в целые
        weights_df['layer'] = weights_df['layer'].astype(int)
        weights_df['annid'] = weights_df['annid'].astype(int)

        #print(weights_df)
        weights_df.to_csv(os.path.join(path, 'weights' + str(begin)+ '.csv'), sep='\t', index=False)

        #---------------------------------------------------------------------------------------------
        # запись весов между слоями для разных эпох        
        for epoch_fs in EPOCHS_FOR_SAMPLING:            
            weights_df_dict[epoch_fs]['weight'] = weights_data_dict[epoch_fs]['weights_col_all']#weights_col_all
            weights_df_dict[epoch_fs]['layer']  = weights_data_dict[epoch_fs][  'layer_col_all']#layer_col_all
            weights_df_dict[epoch_fs]['annid']  = weights_data_dict[epoch_fs][  'annid_col_all']          
            
            # конвертация из вещественных в целые
            weights_df_dict[epoch_fs]['layer'] = weights_df_dict[epoch_fs]['layer'].astype(int)
            weights_df_dict[epoch_fs]['annid'] = weights_df_dict[epoch_fs]['annid'].astype(int)

            #print(weights_df)
            weights_df_dict[epoch_fs].to_csv(os.path.join(path, 'epoch' + str(epoch_fs) + 'weights' + str(begin) + '.csv'), sep='\t', index=False)
        #---------------------------------------------------------------------------------------------



    # ========================================
    # обучили и сохранили модели
    if with_return == True:
        return models, histories
    else:
        del models
        del histories
    


# ============================================
# обучение сетей для энергии и для радиуса
if __name__ == '__main__':
    freeze_support()
    start = time.perf_counter()
        
# обучаемые модели для предсказаний радиуса и энергии
    
models_e    = [] 
histories_e = [] 

models_r    = []
histories_r = [] 


def parallel_training(number_of_threads, path, numofnns, x, y, splitting, LR, LRD, NE, BS, custom_activation, neurons, with_return = False, scale_ymax = None, scale_ymin = None, scale_max = None, scale_min = None, sample_weights_fit = None): # распараллеленное обучение сеток
    print(f'parallel_training: number of threads = {number_of_threads}')    
    print(f'parallel_training: number of ANNs = {numofnns}')  
    models    = [] # список для хранения моделей
    histories = []
    
    #number_of_threads = multiprocessing.cpu_count() # определение числа потоков
    per_thread = numofnns // number_of_threads # число сетей, которое будет приходиться на один поток (не считая остатка)
        
    threads = [] # список для хранения потоков    
    threading_results = [] # список для хранения результатов   
        
    with concurrent.futures.ProcessPoolExecutor(max_workers = number_of_threads) as executor:  
        for i in range(number_of_threads - 1): # инициализация потоков, кроме последнего             
            threads.append(executor.submit(trainsave, path, per_thread, per_thread * i, x, y, splitting, LR, LRD, NE, BS, custom_activation, neurons, with_return = False, scale_ymax = scale_ymax, scale_ymin = scale_ymin, scale_max = scale_max, scale_min = scale_min, sample_weights_fit = sample_weights_fit))       
        
        # submit() method schedules each task              
        #последний поток, если нацело не делится, включает обучение per_thread + остаток от деления сетей
        threads.append(executor.submit(trainsave, path, per_thread + numofnns % number_of_threads, (number_of_threads-1)*per_thread, x, y, splitting, LR, LRD, NE, BS, custom_activation, neurons, scale_ymax = scale_ymax, scale_ymin = scale_ymin, scale_max = scale_max, scale_min = scale_min, sample_weights_fit = sample_weights_fit))
    
    if with_return == True:                        
        for i in range(number_of_threads):
            threading_results.append(threads[i].result())
                
        
        # внесение моделей и историй в списки
        for threading_result in threading_results: 
            models.extend(threading_result[0]) # из кортежа 
            histories.extend(threading_result[1])
        
        return models, histories

#=====================================================
if __name__ == '__main__':
    freeze_support()
    
    summary.write(f'\ntraining_threading = {training_threading}\nnumber of threads = {number_of_threads}\ntraining_threading_2 = {training_threading_2}')  
    
    if training_threading == True:
        # если обучаемых сетей мало, то разделение на два потока + если есть разбивка на тренировочный и валидационный наборы
        if (numofnns_e / number_of_threads) < 1 or (numofnns_r / number_of_threads) < 1 or training_threading_2 == True:
            print('two threads')
            summary.write('\ntrain & save: two threads')           
            
            with concurrent.futures.ProcessPoolExecutor(max_workers = 2) as executor:   
                f1 = executor.submit(trainsave, path_e, numofnns_e, 0, x_e, y_e, splitting, LR_e, LRD_e, NE_e, BS_e, custom_activation_e, neurons_e, 
                                     scale_ymax = df_e.Eabs.max(), scale_ymin = df_e.Eabs.min(),  scale_max = scaler_out_max_e, scale_min = scaler_out_min_e, sample_weights_fit = sample_weights_e)
                f2 = executor.submit(trainsave, path_r, numofnns_r, 0, x_r, y_r, splitting, LR_r, LRD_r, NE_r, BS_r, custom_activation_r, neurons_r, 
                                     scale_ymax = df_r.RMS.max(),  scale_ymin = df_r.RMS.min(),   scale_max = scaler_out_max_r, scale_min = scaler_out_min_r, sample_weights_fit = sample_weights_r)
                
                #models_e, histories_e = f1.result()
                #models_r, histories_r = f2.result()              
                        
                                   
        
        # а если много, то разветвление на много потоков
        else:
            print(f'many threads: {number_of_threads}')
            summary.write('\ntrain & save: many threads')
            print('\nEnergy training...\n')
            pt_e_start = time.perf_counter()            
            
            #models_e, histories_e = parallel_training(number_of_threads, path_e, numofnns_e, x_e, y_e, splitting, LR_e, LRD_e, NE_e, BS_e, custom_activation_e)
            parallel_training(number_of_threads, path_e, numofnns_e, x_e, y_e, splitting, LR_e, LRD_e, NE_e, BS_e, custom_activation_e, neurons_e, 
                              scale_ymax = df_e.Eabs.max(), scale_ymin = df_e.Eabs.min(), scale_max = scaler_out_max_e, scale_min = scaler_out_min_e, sample_weights_fit = sample_weights_e)
            
            pt_e_finish = time.perf_counter()
            summary.write(f'\nEnergy: Training and saving of {numofnns_e} networks finished in {round(pt_e_finish - pt_e_start, 2)} second(s). {round(numofnns_e * 3600/(pt_e_finish - pt_e_start), 1)} trained networks per hour')
            
            print('\nRadius training...\n')
            pt_r_start = time.perf_counter() 
            
            #models_r, histories_r = parallel_training(number_of_threads, path_r, numofnns_r, x_r, y_r, splitting, LR_r, LRD_r, NE_r, BS_r, custom_activation_r)
            parallel_training(number_of_threads, path_r, numofnns_r, x_r, y_r, splitting, LR_r, LRD_r, NE_r, BS_r, custom_activation_r, neurons_r, 
                              scale_ymax = df_r.RMS.max(), scale_ymin = df_r.RMS.min(), scale_max = scaler_out_max_r, scale_min = scaler_out_min_r, sample_weights_fit = sample_weights_r)
            
            pt_r_finish = time.perf_counter()
            summary.write(f'\nRadius: Training and saving of {numofnns_r} networks finished in {round(pt_r_finish - pt_r_start, 2)} second(s). {round(numofnns_r * 3600/(pt_r_finish - pt_r_start), 1)} trained networks per hour')
    
    else:
        print('\nsingle thread\n')
        summary.write('\ntrain & save: no threading')
        print('\nEnergy training...\n')
        models_e, histories_e = trainsave(path_e, numofnns_e, 0, x_e, y_e, splitting, LR_e, LRD_e, NE_e, BS_e, custom_activation_e, neurons_e, with_return = True, 
                                          scale_ymax = df_e.Eabs.max(), scale_ymin = df_e.Eabs.min(), scale_max = scaler_out_max_e, scale_min = scaler_out_min_e, sample_weights_fit = sample_weights_e)
        
        print('\nRadius training...\n')
        models_r, histories_r = trainsave(path_r, numofnns_r, 0, x_r, y_r, splitting, LR_r, LRD_r, NE_r, BS_r, custom_activation_r, neurons_r, with_return = True, 
                                          scale_ymax = df_r.RMS.max(), scale_ymin = df_r.RMS.min(), scale_max = scaler_out_max_r, scale_min = scaler_out_min_r, sample_weights_fit = sample_weights_r)
        
        print('len models_e = ', len(models_e))
        # m_e, h_e = trainsave(path_e, numofnns_e, 0, x_e, y_e, splitting, LR_e, LRD_e, NE_e, BS_e, custom_activation_e)
        # print('\nRadius training...\n')
        # m_r, h_r = trainsave(path_r, numofnns_r, 0, x_r, y_r, splitting, LR_r, LRD_r, NE_r, BS_r, custom_activation_r)
        
    
    
    finish = time.perf_counter()
    print(f'Training finished in {round(finish - start, 2)} second(s)')    
    summary.write(f'\nTraining and saving of {numofnns_e + numofnns_r} networks finished in {round(finish - start, 2)} second(s). {round((numofnns_e + numofnns_r)*3600/(finish - start), 1)} trained networks per hour')
    
    
    print('main: MODELS LEN = ', len(models_e))




# #==================================================================
# данные о loss
if __name__ == '__main__':
    freeze_support()

    # объединение лоссов в один файл
    if splitting == True:        
        assemble_loss_from_traisave_pieces(path_e, 'loss')
        assemble_loss_from_traisave_pieces(path_r, 'loss')
        
        assemble_loss_from_traisave_pieces(path_e, 'val_loss')
        assemble_loss_from_traisave_pieces(path_r, 'val_loss')       
            
    if splitting == False:        
        assemble_loss_from_traisave_pieces(path_e, 'loss')
        assemble_loss_from_traisave_pieces(path_r, 'loss')

    # объединение mse по эпохам в один файл
    assemble_mse_over_epochs_from_traisave_pieces(path_e, 'mse_over_epochs')
    assemble_mse_over_epochs_from_traisave_pieces(path_r, 'mse_over_epochs')  

    # фильтрация mse по эпохам
    filter_mse_over_epochs(path_e, 'mse_over_epochs')
    filter_mse_over_epochs(path_r, 'mse_over_epochs')

    if EXTRA_FITTING == True:
        assemble_mse_over_epochs_from_traisave_pieces(path_e, 'mse_over_extra_epochs')
        assemble_mse_over_epochs_from_traisave_pieces(path_r, 'mse_over_extra_epochs')

    




    # объединение весов в один файл
    if WRITE_WEIGHTS == True:
        assemble_weights_from_traisave_pieces(path_e, 'weights')
        assemble_weights_from_traisave_pieces(path_r, 'weights')

        # объединение весов из частей-расчетов в один файл для каждого epoch_fs
        for epoch_fs in EPOCHS_FOR_SAMPLING:
            assemble_weights_from_traisave_pieces(path_e, f'epoch{epoch_fs}weights')
            assemble_weights_from_traisave_pieces(path_r, f'epoch{epoch_fs}weights')

    

        

    
    # получение и сохранение предсказаний
    print('\nSTART MARKING PREDICTIONS\n')
    start_predictions = time.perf_counter()
    
    if predictions_threading == True:    
        summary.write('\nget_predictions: two threads')
               
        with concurrent.futures.ProcessPoolExecutor(max_workers = 2) as predictions_executor:            
            
            futures = []
            print('models length = ', len(models_e))
                        
            futures.append(predictions_executor.submit(get_predictions_using_path, number_of_threads, df_e, path_e, numofnns_e, 2, n_e, scalerx_e, scalery_e)) # типа поток для энергии
            futures.append(predictions_executor.submit(get_predictions_using_path, number_of_threads, df_r, path_r, numofnns_r, 2, n_r, scalerx_r, scalery_r)) # типа для радиуса
            
            print('SUBMITTING DONE')
            print('futures length = ', len(futures))
            
            preds_e, models_loading_time_e = futures[0].result()
            preds_r, models_loading_time_r = futures[1].result()

        models_loading_time_e = float(models_loading_time_e.total_seconds())
        models_loading_time_r = float(models_loading_time_r.total_seconds())
        
        #print('preds_e = \n', preds_e)
        #print('preds_r = \n', preds_r)
                    
                
            
    else:
        if training_threading == False: # если так, то модели по идее есть в памяти
            print('\nget_predictions: no threading, без загрузки из постоянной памяти')
            summary.write('\nget_predictions: no threading,  без загрузки из постоянной памяти')
            preds_e = get_predictions(df_e, models_e, 2, n_e, scalerx_e, scalery_e)
            preds_r = get_predictions(df_r, models_r, 2, n_r, scalerx_r, scalery_r)
            #data, models, maxN, n, scalerx, scalery
            
        if training_threading == True: # а если обучение параллелилось, то модели надо загружать из постоянной памяти
            print('\nget_predictions: no threading, с загрузкой из постоянной памяти')
            summary.write('\nget_predictions: no threading, с загрузкой из постоянной памяти')
            preds_e, models_loading_time_e = get_predictions_using_path(number_of_threads, df_e, path_e, numofnns_e, 2, n_e, scalerx_e, scalery_e, no_threaded_loading = True)
            preds_r, models_loading_time_r = get_predictions_using_path(number_of_threads, df_r, path_r, numofnns_r, 2, n_r, scalerx_r, scalery_r, no_threaded_loading = True)

            models_loading_time_e = float(models_loading_time_e.total_seconds())
            models_loading_time_r = float(models_loading_time_r.total_seconds())
            
        
        
    
    preds_e.hOmega.round(2)
    preds_r.hOmega.round(2)

    preds_e = preds_e.astype({"Nmax": int})
    preds_r = preds_r.astype({"Nmax": int})
    
    preds_e.to_csv(os.path.join(path_e, 'predictions.csv'), sep='\t', index=False, encoding = 'utf-8')
    preds_r.to_csv(os.path.join(path_r, 'predictions.csv'), sep='\t', index=False, encoding = 'utf-8')
    finish_predictions = time.perf_counter()
    summary.write(f'\nGetting predictions of {numofnns_e + numofnns_r} networks finished in {round(finish_predictions - start_predictions, 2)} second(s). {round((len(preds_e) + len(preds_r))*60/(finish_predictions - start_predictions), 1)} lines per minute ')

    
    print('PREDICTIONS DONE')
    
    # считывание предсказаний из файла
    # path_e = pathlib.Path('G:/Мой диск/!ML/E_or_R_only/multiproc/Nplot_li/test_03_04_2022_03-36/energy')    
    # path_r = pathlib.Path('G:\Мой диск/!ML\E_or_R_only/multiproc/Nplot_li/test_03_04_2022_03-36/radius')
    # preds_e = pd.read_csv(os.path.join(path_e, 'predictions.csv'), sep='\t', index_col=False)
    # print(preds_e)
    # print(len(preds_e['annid'].unique()))
    
    # preds_r = pd.read_csv(os.path.join(path_r, 'predictions.csv'), sep='\t', index_col=False)
    # print(preds_r)
    # print(len(preds_r['annid'].unique()))
    # ============================================

    
    
    t_end = datetime.datetime.now()
    
    if training_threading == True or predictions_threading == True:
        summary.write('\nModels loading time:')
        summary.write(f'\n{round(models_loading_time_e + models_loading_time_r,1)} seconds, {round((numofnns_e + numofnns_r) / (models_loading_time_e + models_loading_time_r), 1)} models per second')
        summary.write(f'\nEnergy: {models_loading_time_e}, {round(numofnns_e / models_loading_time_r, 1)} models per second')
        summary.write(f'\nRadius: {models_loading_time_r}, {round(numofnns_r / models_loading_time_r, 1)} models per second')
    
    
    summary.write(f'\nEND\t{t_end}')
    print(f'Finished in {(t_end - t_begin).total_seconds()} second(s)')    
    summary.write(f'\nFinished in {(t_end - t_begin).total_seconds()} second(s). {round((numofnns_e + numofnns_r)*3600/float((t_end - t_begin).total_seconds()),1)} networks per hour in average')
    summary.close()
    
    
    os.system("pause")
    os._exit(0)