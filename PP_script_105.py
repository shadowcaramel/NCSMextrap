from concurrent.futures import thread
import configparser
import pathlib
import os
import datetime
import time
import re
import pandas as pd
import concurrent.futures
import multiprocessing
from multiprocessing import freeze_support
# import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
import statistics
from scipy.stats import norm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
#import seaborn as sns
#from labellines import labelLine, labelLines
# import pydot
import math
import tensorflow as tf
import tensorflow_addons as tfa
import keras
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense
#from keras.utils.generic_utils import get_custom_objects
#from tensorflow.keras.layers import Activation
import threading
import concurrent.futures
import multiprocessing, logging
from multiprocessing import freeze_support
from supervenn import supervenn # для диаграмм множеств номеров сеток
import imageio.v3 as iio # для создания гифок из гистограмм весов
from pygifsicle import optimize as pgs_optimize # для оптимизации сгенерированных гифок





PP_summary = open('PP_summary.txt', 'w', encoding = 'utf8') # отчет о пост-процессинге
PP_begin = datetime.datetime.now()
PP_summary.write(f'Post-processing BEGIN: {PP_begin}\n\n')

# чтение конфиг-файла
cg = configparser.ConfigParser()
cg.read(pathlib.Path.cwd() / '..' /'config.ini', encoding = 'utf8')

Notes = cg['Notes']['Notes']


#================================
# некоторые параметры Slurm
PP_ntasks = cg['Slurm parameters'].getint('PP_ntasks')

#================================    
# пути к исходным данным 
paths_to_the_data = cg['Paths to the data']    
path_data_e    = pathlib.Path(paths_to_the_data['path_energy'])    
path_data_r    = pathlib.Path(paths_to_the_data['path_radius'])
columns_data_e = paths_to_the_data['columns_energy']
columns_data_r = paths_to_the_data['columns_radius']


#================================
NUM_PARTS = cg['Job parallelism'].getint('script_parts')


#================================
# препроцессинг данных
    # логические
cut_horizontal_EandR  = cg['Data pre-processing logical parameters'].getboolean('cut_horizontal_e_and_r')    

    # для энергии
data_prep_e = cg['Data pre-processing parameters for E']
cut_on_min              = data_prep_e.getboolean('cut_on_min')
horizontal_cut          = data_prep_e.getboolean('horizontal_cut')    
minN_e                  = data_prep_e.getint('min_Nmax')
maxN_e                  = data_prep_e.getint('max_Nmax')
hwcutoff_l_e            = data_prep_e.getfloat('hwcutoff_left')
hwcutoff_r_e            = data_prep_e.getfloat('hwcutoff_right')
hwcutoff_l_e_forExtrapB = data_prep_e.getfloat('hwcutoff_left_for_Extrap_B')
E_horizontal            = data_prep_e.getfloat('E_horizontal')
Nmax_for_metric_e       = data_prep_e.getint('Nmax_for_metric')
scaler_in_min_e         = data_prep_e['scaler_in_min']
scaler_in_max_e         = data_prep_e['scaler_in_max']
scaler_out_min_e        = data_prep_e.getfloat('scaler_out_min')
scaler_out_max_e        = data_prep_e.getfloat('scaler_out_max')

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
splitting     = gen_train_params.getboolean('splitting')
bias_usage    = gen_train_params.getboolean('bias_usage')
EXTRA_FITTING = gen_train_params.getboolean('Extra_fitting')
SHUFFLE       = gen_train_params.getboolean('shuffle')
SHUFFLE_EXTRA = gen_train_params.getboolean('shuffle_exta')
WRITE_WEIGHTS = gen_train_params.getboolean('write_weights')


#================================
# числовые настройки обучения
    # для энергии
num_train_params_e = cg['Numeric training parameters for E']
numofnns_e = num_train_params_e.getint('num_of_neural_networks')

    
    # для радиуса    
num_train_params_r = cg['Numeric training parameters for R']
numofnns_r = num_train_params_r.getint('num_of_neural_networks')   


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
# общие параметры пост-процессинга
pp_general_params = cg['General post-processing parameters']
Deviation_quantile                  = pp_general_params.getfloat('deviation_quantile')
plotting_threading                  = pp_general_params.getboolean('plotting_threading')
loss_hist_bins                      = pp_general_params.getint('loss_histograms_bins')
PREDICTIONS_FOR_NMAX_STEP           = pp_general_params.getint('predictions_for_nmax_step')
PREDICTIONS_FOR_HOMEGA_MINIMAL_STEP = pp_general_params.getfloat('predictions_for_hOmega_minimal_step')
LOSS_SELECTION_QUANTILE             = pp_general_params.getfloat('loss_selection_quantile')

iterative_outlier_filtering_method  = pp_general_params.get('iterative_outlier_filtering_method')
N_SIGMA_SELECTION                   = pp_general_params.getint('n_sigma_selection')
BOXPLOT_RULE_COEF                   = pp_general_params.getfloat('boxplot_rule_coef')


#=================================
# параметры пост-процессинга для энергии
pp_params_e = cg['Post-processing parameters for E']
MIN_MODELS_e          = pp_params_e.getint('min_number_of_models')
E_exact               = pp_params_e['exact_value']
plotting_limits_min_e = pp_params_e.getfloat('plotting_limits_min')
plotting_limits_max_e = pp_params_e.getfloat('plotting_limits_max')
#вариационный принип >>>
n_variational_e       = pp_params_e.getint('max_Nmax_for_variational_principle')
VP_HO_LEFT            = pp_params_e.getfloat(  'VP_hO_left')
VP_HO_RIGHT           = pp_params_e.getfloat(  'VP_hO_right')
VP_HO_MINIMAL_STEP    = pp_params_e.getfloat(  'VP_hO_minimal_step')
VP_EPSILON            = pp_params_e.getfloat(  'VP_epsilon')
VP_ADDITIONAL_CHECK   = pp_params_e.getboolean('VP_additional_check')
VP_EPSILON_2          = pp_params_e.getfloat(  'VP_epsilon_2')
VP_CHECK_MODE         = pp_params_e.getint(    'VP_check_mode')
# <<<
# сходимость >>>
EPS_HO                = pp_params_e.getfloat('eps_ho')
EPS_NMAX              = pp_params_e.getfloat('eps_nmax')
CONV_LEFT_QUANTILE    = pp_params_e.getfloat('convergence_left_quantile')
CONV_RIGHT_QUANTILE   = pp_params_e.getfloat('convergence_right_quantile')
# <<<
STRAIGHTNESS_TOLERANCE_e    = pp_params_e.getfloat('straightness_tolerance')

if  E_exact == 'None':
    E_exact = None
else:
    E_exact = pp_params_e.getfloat('exact_value')




#=================================
# параметры пост-процессинга для радиуса
pp_params_r = cg['Post-processing parameters for R']
n_variational_r = pp_params_r.getint('max_Nmax_for_variational_principle')
MIN_MODELS_r = pp_params_r.getint('min_number_of_models')
rms_exact = pp_params_r['exact_value']
plotting_limits_min_r = pp_params_r.getfloat('plotting_limits_min')
plotting_limits_max_r = pp_params_r.getfloat('plotting_limits_max')
STRAIGHTNESS_TOLERANCE_r    = pp_params_r.getfloat('straightness_tolerance')


if  rms_exact == 'None':
    rms_exact = None
else:
    rms_exact = pp_params_r.getfloat('exact_value')


#================================
# общие логические настройки обучения
gen_train_params = cg['General training parameters']
splitting     = gen_train_params.getboolean('splitting')
bias_usage    = gen_train_params.getboolean('bias_usage')
EXTRA_FITTING = gen_train_params.getboolean('Extra_fitting')
SHUFFLE       = gen_train_params.getboolean('shuffle')
SHUFFLE_EXTRA = gen_train_params.getboolean('shuffle_exta')


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
#--------------------------------




EPOCHS_FOR_SAMPLING = sorted(list(set(np.geomspace(1, min(NE_e, NE_r) - 1, num = NUM_OF_EPOCH_SAMPLES_FOR_HISTS, endpoint=True, dtype = int))))

#=================================
# создание папок с картинками
path_e = pathlib.Path.cwd() / 'energy'
path_r = pathlib.Path.cwd() / 'radius'
pics_path_e = path_e / 'pics' / 'energy' # путь для сохранения картинок касательно энергии
pics_path_r = path_r / 'pics' / 'radius' # касательно энергии

# создание нужных папок 
pathlib.Path(pics_path_e).mkdir(parents=True, exist_ok=True) 
pathlib.Path(pics_path_r).mkdir(parents=True, exist_ok=True)


plt.rc('figure',  figsize   = (figsize_width, figsize_height))  # setting default size of plots
plt.rc('axes'  ,  titlesize = axes_titlesize)   # fontsize of the axes title
plt.rc('axes'  ,  labelsize = axes_labelsize)   # fontsize of the x and y labels
plt.rc('xtick' ,  labelsize = xtick_labelsize)  # fontsize of the tick labels
plt.rc('ytick' ,  labelsize = ytick_labelsize)

# границы для графиков
plotting_limits = [plotting_limits_min_e, plotting_limits_max_e], [plotting_limits_min_r, plotting_limits_max_r]

number_of_threads = PP_ntasks


def assemble_loss_or_predictions_from_partial_jobs(data_name, value_name, path):
    # data_name - название файла: predictions, loss или val_loss или че-то ещё
    # value_name - энергия или радиус
    # объединение предсказаний, loss'ов, сгенерированных в задачах-частях в один файл 
    # для файла с весами тоже подходит
    print(f'assembling {value_name} from {data_name}: {path}...')
    assembled = pd.DataFrame()
    
    partial_dataframes = []
    for part in range(NUM_PARTS):                    
        partial_data = pd.read_csv(pathlib.Path.cwd() / '..' / f'part_{part}' / value_name / (data_name + '.csv'), sep = '\t')        
        numofnns = len(partial_data['annid'].unique())
        partial_data.annid += int(part * numofnns)   
        partial_dataframes.append(partial_data)

    assembled = pd.concat(partial_dataframes, ignore_index=True)
    #assembled.sort_values(by=['annid'])
      
    #path_to_place = pathlib.Path(path).mkdir(parents=True, exist_ok=True) 

    assembled.to_csv((path / f'{data_name}.csv'), sep='\t', index=False, float_format = '%.5e')
    print(f'{value_name} from {data_name}: {path} assembled') 

def Metric(deviation, scatter, a, b): # D^a * Sf^b Метрика, характеризующая предсказания: (отклонение на некоторой выборке) * (разброс предсказаний)
    return math.pow(deviation, a) * math.pow(scatter, b)

def cut_left_than_minE(df):
    # обрезание энергии левее минимума
    minN = df['Nmax'].min()
    #print(minN)
    maxN = df['Nmax'].max()
    #print(maxN)
    Nmaxes = df.Nmax.unique()
    for Nmax in Nmaxes:
        #print(Nmax)
        minE_idx = df[df['Nmax'] == Nmax]['Eabs'].idxmin()
        #print(minE_idx)
        corr_hO = df[df['Nmax'] == Nmax]['hOmega'][minE_idx]
        #print(corr_hO)
        df.drop(df[df['Nmax'] == Nmax][df.hOmega < corr_hO].index, inplace=True)

def energyplot(preds_energy, preds_radius, maxN, n_energy, n_radius, variational_min, plot_title, ax1=None, exact_value=None, extrapolation_value=None, plot_limits=None, bins=None):
    # построение графика минимума энергии для разных сетей
    # из-за переделывания требуются данные как для энергии, так и для радиуса
    # да, тупо
    # ============================================
    if ax1 is None:
        ax1 = plt.gca()

    avgstd = avg_and_std(preds_energy, preds_radius, n_energy, n_radius)
    stdE = float(avgstd.stdE.astype(float))
    avgE = float(avgstd.avgE.astype(float))

    medE =         float(avgstd.medE)
    midspreadE =   float(avgstd.midspreadE)  # Q3-Q1
    EQ1 = float(avgstd.EQ1)
    EQ3 = float(avgstd.EQ3)

    plt.rc('figure',  figsize   = (figsize_width, figsize_height))  # setting default size of plots
    plt.rc('axes'  ,  titlesize = axes_titlesize)   # fontsize of the axes title
    plt.rc('axes'  ,  labelsize = axes_labelsize)   # fontsize of the x and y labels
    plt.rc('xtick' ,  labelsize = xtick_labelsize)  # fontsize of the tick labels
    plt.rc('ytick' ,  labelsize = ytick_labelsize)
    Ns = np.array(list(range(maxN + 2, n_energy + 1, 2)))
    # print(Ns)    
    
    
    ids = preds_energy['annid'].unique()  
    #print(ids)
       
    
    for i in ids:  # цикл по сеткам
        mins = []
        EN = preds_energy[['Eabs', 'Nmax']].loc[preds_energy['annid'] == i]               
        for N in range(maxN + 2, n_energy + 1, 2):  # цикл как бы по Nmax
            #Emin = preds['E'][preds['annid'] == i][preds['Nmax'] == N]
            Emin = EN[['Eabs']].loc[EN['Nmax'] == N]
            # plt.plot(N, Emin.min())
            mins.append(Emin.min())
            #print(Emin['E'].min())
            # print(Emin.min())
        # print(mins)
        mins = np.array(mins)
        # print(mins)
        
        ax1.plot(Ns, mins, label=i)

    #labelLines(plt.gca().get_lines(), align=True, fontsize=10, color = 'k')
    ids = preds_energy['annid'].unique()  # номера нейросетей, которые остались
    number_of_nns = len(ids)  # число сетей

    plot_title = plot_title + ', ' + r'$E = ' + str(format(avgE, '.3f')) + '\pm' + str(format(stdE, '.3f')) + "$; " + 'сетей: ' + str(number_of_nns) + \
                            '\n' + f'median = {medE: .3f}, midspread = {midspreadE: .3f}' + '\n' + f'Q1 = {EQ1: .3f}, Q3 = {EQ3: .3f}'

    plt.ylabel('$min(E)$')
    plt.xlabel('$N_{max}$')

    if exact_value is not None:
        plt.axhline(y=exact_value, color='black', linestyle='-', linewidth=4)  # точное значение
    plt.axhline(y=variational_min, color='#5736f7', linestyle='-', linewidth=4)  # вариационный минимум

    if extrapolation_value is not None:  # значение энергии, полученное экстраполяцией
        plt.axhline(y=extrapolation_value, color='forestgreen', linestyle='--', linewidth=4)

    plt.axhline(y=avgE, color='tomato', linestyle='-', linewidth=4)  # среднее значение предсказаний

    # Q1, Q3 и медиана
    plt.axhline(y = EQ1, color = 'magenta', linestyle='--', linewidth = 2, alpha = 0.5)
    plt.axhline(y = EQ3, color = 'magenta', linestyle='--', linewidth = 2, alpha = 0.5)
    plt.axhline(y = medE, color = 'magenta', linestyle='-', linewidth = 4, alpha = 0.5)

    ax1.axhspan(avgE - stdE, avgE + stdE, facecolor='tomato', alpha=0.1)  # закрашивание разброса величины

    plt.title(plot_title, fontsize=30)

    plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
    plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f'))
    
    if plot_limits is not None:
        ax1.set_ylim(plot_limits)        
        
    if bins is not None:        
        ax1.yaxis.set_major_locator(plt.MaxNLocator(bins, prune = None))    
        
        
        

    # plt.legend()
    return (ax1)

def rmsplot(preds_energy, preds_radius, maxN, n_energy, n_radius, plot_title, ax=None, exact_value=None, plot_limits=None, bins=None):
    # из-за переделывания требуются данные как для энергии, так и для радиуса
    # да, тупо
    #print(f'n_radius = {n_radius}')
    
    if ax is None:
        ax = plt.gca()

    plt.ylabel('$RMS$')
    plt.xlabel('$\hbar \Omega$')

    plt.rc('axes', titlesize=20)  # fontsize of the axes title
    plt.rc('axes', labelsize=30)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=30)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=30)

    # hs = list(range(hwcutoff_l, hwcutoff_r + 1, ))
    
    hs = preds_radius['hOmega'].unique()
    hs.sort()    
    #print(f'hOmegas = {hs}')
    
    ids = preds_radius['annid'].unique()  # номера нейросетей, которые остались
    ids.sort()    
    #print(f'ids = {ids}')
    
    
    
    rms = preds_radius[preds_radius['Nmax'] == n_radius]    
    #print(rms)
    

    for i in ids:
        #rms = preds_radius['RMS'][preds_radius['annid'] == i][preds_radius['Nmax'] == n_radius]
        rms_forplot = rms['RMS'].loc[rms['annid'] == i]                 
        #print(len(rms_forplot))   
        #print(len(hs)) 
        plt.plot(hs, rms_forplot, label=i)

    if exact_value is not None:
        plt.axhline(y=exact_value, color='black', linestyle='-', linewidth=4)
        
    avgstd = avg_and_std(preds_energy, preds_radius, n_energy, n_radius)
    avgRMS = float(avgstd.avgRMS.astype(float))
    stdRMS = float(avgstd.stdRMS.astype(float))

    medRMS =         float(avgstd.medRMS)
    midspreadRMS =   float(avgstd.midspreadRMS)  # Q3-Q1
    RMSQ1 = float(avgstd.RMSQ1)
    RMSQ3 = float(avgstd.RMSQ3)

    plt.axhline(y=avgRMS, color='gold', linestyle='-', linewidth=4)  # среднее значение предсказаний

    ax.axhspan(avgRMS - stdRMS, avgRMS + stdRMS, facecolor='gold', alpha=0.1)  # закрашивание разброса величины

    plt.axhline(y = RMSQ1, color = 'magenta', linestyle='--', linewidth = 2, alpha = 0.5)
    plt.axhline(y = RMSQ3, color = 'magenta', linestyle='--', linewidth = 2, alpha = 0.5)
    plt.axhline(y = medRMS, color = 'magenta', linestyle='-', linewidth = 4, alpha = 0.5)

    number_of_nns = len(ids)  # число сетей

    plot_title = plot_title + ', ' + r'$RMS = ' + str(format(avgRMS, '.3f')) + '\pm' + \
                 str(format(stdRMS, '.3f')) + "$; " + 'сетей: ' + str(number_of_nns) + \
                 '\n' + f'median = {medRMS: .3f}, midspread = {midspreadRMS: .3f}' + '\n' + f'Q1 = {RMSQ1: .3f}, Q3 = {RMSQ3: .3f}'

    plt.title(plot_title, fontsize=30)

    #labelLines(plt.gca().get_lines(), align=False, fontsize=10, color='k')

    # plt.legend()
    plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
    plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f'))
    
    if plot_limits is not None:
        ax.set_ylim(plot_limits)        
        
    if bins is not None:        
        ax.yaxis.set_major_locator(plt.MaxNLocator(bins, prune = None))
    
    return (ax)

def extrapolation_b(data):
    # экстраполяция B по трем соседним модельным пространствам
    # возвращается минимум

    # print(data)
    
    # плюс то же самое, но с предыдущим модельным пространством для оценки погрещности 
    maxes = list(data['Nmax'].unique())
    maxes.sort(reverse=True)  # сортировка по убыванию
    
    maxes_larger = maxes[:3]  # первых три самых больших модельных пространства
    maxes_lower  = maxes[1:4] # следующие три (сдвиг на одно мод. пр-вр) самых больших мод. пр-ва
    
    # print(maxs)
    def extrapB(data, maxs):
        EN_plus = data[['Eabs', 'hOmega', 'Nmax']][data.Nmax == maxs[0]]  # старшее модельное пространство
        # print(EN_plus)
        EN_plus_min_hO = EN_plus.hOmega.min()
        EN_plus_max_hO = EN_plus.hOmega.max()

        EN = data[['Eabs', 'hOmega', 'Nmax']][data.Nmax == maxs[1]]  # среднее модельное пространство
        # print(EN)
        EN_min_hO = EN.hOmega.min()
        EN_max_hO = EN.hOmega.max()

        EN_minus = data[['Eabs', 'hOmega', 'Nmax']][data.Nmax == maxs[2]]  # младшее модельное пространство
        # print(EN_minus)
        EN_minus_min_hO = EN_minus.hOmega.min()
        EN_minus_max_hO = EN_minus.hOmega.max()
        
        # print(EN_plus)
        # print(EN)
        # print(EN_minus)
        #print(EN_minus.loc[2545:2750])

        hO_min = max([EN_plus_min_hO, EN_min_hO,
                    EN_minus_min_hO])  # минимальное hOmega, которое встречается среди всех трех моделльных пространствах
        hO_max = min([EN_plus_max_hO, EN_max_hO,
                    EN_minus_max_hO])  # максимальное hOmega, которое встречается среди всех трех моделльных пространствах
        hO_min = round(hO_min, 2)
        hO_max = round(hO_max, 2)
        
        # print(hO_min)
        # print(hO_max)

        # унифицируем по hOmega
        EN_plus.drop(EN_plus[EN_plus.hOmega < hO_min].index,     inplace=True)
        EN.drop(EN[EN.hOmega < hO_min].index,                    inplace=True)
        EN_minus.drop(EN_minus[EN_minus.hOmega < hO_min].index,  inplace=True)

        EN_plus.drop(EN_plus[EN_plus.hOmega > hO_max].index,     inplace=True)
        EN.drop(EN[EN.hOmega > hO_max].index,                    inplace=True)
        EN_minus.drop(EN_minus[EN_minus.hOmega > hO_max].index,  inplace=True)

        print(f'ExtrapB: EN_plus = {EN_plus}')
        print(f'ExtrapB: EN = {EN}')
        print(f'ExtrapB: EN_minus = {EN_minus}')
        
        #print(f'ExtrapB: EN_plus = {len(EN_plus)}')
        #print(f'ExtrapB: EN = {len(EN)}')
        #print(f'ExtrapB: EN_minus = {len(EN_minus)}')
        
        # унификация еще раз (например если диапазон одинаковый, но шаг разный)
        set_hO_EN_plus  = set(EN_plus.hOmega.unique())
        set_hO_EN       = set(EN_plus.hOmega.unique())
        set_hO_EN_minus = set(EN_plus.hOmega.unique())

        common_hO = set_hO_EN_plus.intersection(set_hO_EN, set_hO_EN_minus)
        EN_plus  = EN_plus[ EN_plus[ 'hOmega'].isin(common_hO)]
        EN       = EN[      EN[      'hOmega'].isin(common_hO)]
        EN_minus = EN_minus[EN_minus['hOmega'].isin(common_hO)]



        # print(EN_plus)
        # print(EN)
        #print(EN_minus)

        # print(list(EN.hOmega))

        extrapolation = []  # hOmega и Eabs

        for i in list(EN.hOmega):
            eb_n = float(EN.Eabs[EN.hOmega == i]) - float(EN_minus.Eabs[EN_minus.hOmega == i])  # числитель формулы (3)        
            eb_d = float(EN_plus.Eabs[EN_plus.hOmega == i]) - float(EN.Eabs[EN.hOmega == i])  # знаменатель (3)
            try:
                eb = eb_n / eb_d
            except ZeroDivisionError as Exc:
                print(Exc)
                PP_summary.write(f'Zero Division Error: hOmega = {i}')
                #print(i)                
                return None, None
            

            CN_n = float(EN.Eabs[EN.hOmega == i]) - float(EN_minus.Eabs[EN_minus.hOmega == i])  # числитель формулы (4)
            CN_d = 1 - eb  # знаменатель формулы (4)            
            try:
                CN = CN_n / CN_d
                # print(CN)
            except ZeroDivisionError as Exc:
                print(Exc)
                PP_summary.write(f'Zero Division Error: hOmega = {i}')
                #print(i)
                return None, None
                
            

            E_inf = float(EN.Eabs[EN.hOmega == i]) - CN  # формула (6)
            # print(E_inf)
            extrapolation.append([i, E_inf])

        # print(extrapolation)
        extrapolation = pd.DataFrame(extrapolation, columns=['hOmega', 'E_inf'])
        #print(extrapolation)

        #E_inf_min = extrapolation.E_inf.min()
        # print(E_inf_min)

        plt.rc('figure',  figsize   = (figsize_width, figsize_height))  # setting default size of plots
        plt.rc('axes'  ,  titlesize = axes_titlesize)   # fontsize of the axes title
        plt.rc('axes'  ,  labelsize = axes_labelsize)   # fontsize of the x and y labels
        plt.rc('xtick' ,  labelsize = xtick_labelsize)  # fontsize of the tick labels
        plt.rc('ytick' ,  labelsize = ytick_labelsize)

        plt.plot(extrapolation.hOmega, extrapolation.E_inf, color='forestgreen', linewidth = 4)
        
        lowest_curve = data[['Eabs', 'hOmega', 'Nmax']][data.Nmax == maxs[0]]  # старшее модельное пространство    
        #print(lowest_curve)
        plt.plot(lowest_curve['hOmega'], lowest_curve['Eabs'], color = 'tomato', linewidth = 4) 
        
        #вычисляем расстояние между кривыми при каждой hOmega
        dist = lowest_curve[['hOmega', 'Eabs']]
        dist.reset_index(inplace=True) # без сброса индекса не делается нормального вычитания
        dist['Eabs'] = abs(dist['Eabs'] - extrapolation['E_inf'])
        #print(dist)
        index = dist['Eabs'].idxmin()    
        #print(index)
        E_extrap = extrapolation['E_inf'][index]     
        #print(E_extrap)

        
        
        plt.axhline(y=E_extrap, color='forestgreen', linestyle='--')

        #plot_title = 'Extapolation B ' + str(maxs) + ': ' r'$E_{\infty}^{min} = ' + str(format(E_inf_min, '.3f')) + "$"
        plot_title = 'Extapolation B ' + str(maxs) + ': ' r'$E_{\infty} = ' + str(format(E_extrap, '.3f')) + "$"
        plt.title(plot_title, fontsize=30)

        plt.ylabel('$E$')
        plt.xlabel('$\hbar \Omega$')

        plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
        plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f'))
        plt.savefig(pics_path_e / ("Экстраполяция B" + str(maxs) + "." + plot_format), format = plot_format)
        #plt.show()
        plt.close()

        return extrapolation, index

    # датафреймы, содержащие значения экстраполяции для старших мод. пр-в и индекс "оптимального" hOmega
    E_extrap_larger_df, index_larger = extrapB(data, maxes_larger)
    E_extrap_lower_df,  index_lower  = extrapB(data, maxes_lower)

    # если в процессе вычисления экстраполированного значения возникла ошибка, то возвращается None, и тут делается соотв проверка и возращается None
    if index_larger is None or index_lower is None:
        print('\n !!! an error has occured during ExtrapB calculations !!!')
        PP_summary.write('\n !!! an error has occured during ExtrapB calculations !!!')
        return None, None

    E_extrap_larger_val = E_extrap_larger_df['E_inf'][index_larger] # итоговое значение экстраполяции
    E_extrap_lower_val  = E_extrap_lower_df['E_inf'][index_lower]


    # для оценки погрешности берется разница между экстраполяцией для больших мод. пр-в
    # и значением экстраполяции для меньших модельных пространств при "оптимальном" (для больших мод. пр-в) hO
    extrap_uncertainty = E_extrap_lower_df['E_inf'][index_larger] - E_extrap_larger_val

    # на общий график
    plt.plot(E_extrap_larger_df.hOmega, E_extrap_larger_df.E_inf, color='forestgreen', linewidth = 4)
    plt.plot(E_extrap_lower_df.hOmega,  E_extrap_lower_df.E_inf,  color='limegreen',   linewidth = 4)


    lowest_curve_larger = data[['Eabs', 'hOmega', 'Nmax']][data.Nmax == maxes_larger[0]]  # старшее модельное пространство для старших мод пр-в   
    lowest_curve_lower =  data[['Eabs', 'hOmega', 'Nmax']][data.Nmax == maxes_lower[0]]   # старшее модельное пространство для младших мод пр-в   
    #print(lowest_curve)
    plt.plot(lowest_curve_larger['hOmega'], lowest_curve_larger['Eabs'], color = 'tomato', linewidth = 4) # старшее
    plt.plot(lowest_curve_lower ['hOmega'], lowest_curve_lower ['Eabs'], color = 'salmon', linewidth = 4) # помладше


    plt.axhline(y=E_extrap_larger_val, color='forestgreen', linestyle='--')
    plot_title = 'Extapolation B ' + str(maxes_larger)  + ': ' r'$E_{\infty} = ' + str(format(E_extrap_larger_val, '.3f')) + "$"
    plot_title += '\n' + str(maxes_lower)  + ': ' r'$E_{\infty} = ' + str(format(E_extrap_lower_val, '.3f')) + "$" + '\n' + f'Extrapolation uncertainty: {extrap_uncertainty: .3f}'
    plt.title(plot_title, fontsize=30)

    plt.ylabel('$E$')
    plt.xlabel('$\hbar \Omega$')

    plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
    plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f'))
    plt.savefig(pics_path_e / ("Экстраполяция B" + str(maxes_larger) + "_merged" + "." + plot_format), format = plot_format)
    #plt.show()
    plt.close()

    return E_extrap_larger_val, extrap_uncertainty

def avg_and_std(preds_energy, preds_radius, n_energy, n_radius):
    emin_annid = []  # список с минимальной энергией для разных сетей
    ravg_annid = []  # список с усредненным по разным hOmega радиусами для разных сетей
    
    
    ids_e = preds_energy['annid'].unique()  # номера нейросетей, которые остались
    ids_r = preds_radius['annid'].unique()  # номера нейросетей, которые остались
    
    # print(len(ids))
    for i in ids_e:  # цикл по сеткам
        Emin = preds_energy['Eabs'][preds_energy['annid'] == i][preds_energy['Nmax'] == n_energy].min()        
        emin_annid.append([Emin, i])
        
    for i in ids_r:  # цикл по сеткам        
        avgRMS = preds_radius['RMS'][preds_radius['annid'] == i][preds_radius['Nmax'] == n_radius].mean()
        ravg_annid.append([avgRMS, i])


    emin_annid = pd.DataFrame(emin_annid, columns=['E', 'annid']) # делаем датафрейм из списка
    ravg_annid = pd.DataFrame(ravg_annid, columns=['RMS', 'annid'])
    

    avgE = emin_annid['E'].mean()  # среднее
    stdE = emin_annid['E'].std()  # стандартное отклонение 

    avgRMS = ravg_annid['RMS'].mean()
    stdRMS = ravg_annid['RMS'].std()

    medE = emin_annid['E'].median()  # медианное значение
    EQ1 = emin_annid['E'].quantile(q=0.25, interpolation='linear')  # первый (нижний) квартиль
    EQ3 = emin_annid['E'].quantile(q=0.75, interpolation='linear')  # третий (верхний) квартиль
    midspreadE = EQ3 - EQ1

    medRMS = ravg_annid['RMS'].median()
    RMSQ1 = ravg_annid['RMS'].quantile(q=0.25, interpolation='linear')  # первый (нижний) квартиль
    RMSQ3 = ravg_annid['RMS'].quantile(q=0.75, interpolation='linear')  # третий (верхний) квартиль
    midspreadRMS = RMSQ3 - RMSQ1

    avg_std = pd.DataFrame([[        avgE,   stdE,   medE,   midspreadE,   EQ1,   EQ3,   avgRMS,   stdRMS,   medRMS,   midspreadRMS,   RMSQ1,   RMSQ3,]], \
                           columns=['avgE', 'stdE', 'medE', 'midspreadE', 'EQ1', 'EQ3', 'avgRMS', 'stdRMS', 'medRMS', 'midspreadRMS', 'RMSQ1', 'RMSQ3',])

    return avg_std

def calculate_statistics(predictions_e, predictions_r, ids_e, ids_r):
    '''
    ввод:
    predictictions - датафрейм с предсказаниями
    ids - список (id) сеток для анализа

    расчет статистических показателей: медианы, IQR, Q1, Q3
    на основе массива предсказаний

    вывод:
    датафрейм со статистическими показателями


    параметры внешние (из конфига)
    n_e - максимальный Nmax для получения предсказаний энергии
    n_r - максимальный Nmax для получения предсказаний энергии
    '''

    #predictions_selected_e = predictions_e[predictions_e['annid'].isin(ids_e)]
    #predictions_selected_r = predictions_r[predictions_r['annid'].isin(ids_r)]

    calc_results_df_e = pd.DataFrame(columns = ['Nmax', 'median', 'IQR', 'Q1', 'Q3', 'count'])
    calc_results_df_r = pd.DataFrame(columns = ['Nmax', 'median', 'IQR', 'Q1', 'Q3', 'count'])     


    emin_annid = []  # список с минимальной энергией для разных сетей
    ravg_annid = []  # список с усредненным по разным hOmega радиусами для разных сетей

    for i in ids_e:  # цикл по сеткам        
        Emin = predictions_e['Eabs'][predictions_e['annid'] == i][predictions_e['Nmax'] == n_e].min()        
        emin_annid.append([Emin, i])
        
    for i in ids_r:  # цикл по сеткам        
        avgRMS = predictions_r['RMS'][predictions_r['annid'] == i][predictions_r['Nmax'] == n_r].mean()
        ravg_annid.append([avgRMS, i])

    

    emin_annid = pd.DataFrame(emin_annid, columns=['E', 'annid']) # делаем датафрейм из списка
    ravg_annid = pd.DataFrame(ravg_annid, columns=['RMS', 'annid']) 

    medE = emin_annid['E'].median()  # медианное значение    
    EQ1 = emin_annid['E'].quantile(q=0.25, interpolation='linear')  # первый (нижний) квартиль
    EQ3 = emin_annid['E'].quantile(q=0.75, interpolation='linear')  # третий (верхний) квартиль
    midspreadE = EQ3 - EQ1

    medRMS = ravg_annid['RMS'].median()
    RMSQ1 = ravg_annid['RMS'].quantile(q=0.25, interpolation='linear')  # первый (нижний) квартиль
    RMSQ3 = ravg_annid['RMS'].quantile(q=0.75, interpolation='linear')  # третий (верхний) квартиль
    midspreadRMS = RMSQ3 - RMSQ1

    calc_results_df_e['Nmax'] = [maxN_e] #максимальный Nmax в обучающей выборке
    calc_results_df_e['median']  = [medE]    
    calc_results_df_e['IQR']     = [midspreadE]
    calc_results_df_e['Q1']      = [EQ1]
    calc_results_df_e['Q3']      = [EQ3]    
    calc_results_df_e['count']   = [len(ids_e)  ]  

    calc_results_df_r['Nmax'] = [maxN_r] #максимальный Nmax в обучающей выборке
    calc_results_df_r['median']  = [medRMS]
    calc_results_df_r['IQR']     = [midspreadRMS]
    calc_results_df_r['Q1']      = [RMSQ1]
    calc_results_df_r['Q3']      = [RMSQ3]
    calc_results_df_r['count']   = [len(ids_r)]

    return calc_results_df_e, calc_results_df_r


def histograms(preds_energy, preds_radius, n_energy, n_radius, plot_title, bins, plot_limits=None, axis_bins=None):  # гистограммы распределения минимума энергии и усредненного радиуса (для большого Nmax)

    # датафрейм с энергии и с усредненным по разным hOmega радиусами для разных сетей
    # тупо, но пофиг
    # ==================================================

    # нормальное распределение
    def pdf(x, sigma, mu):
        return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp( - (x - mu)**2 / (2 * sigma**2) )



    if plot_limits is not None:
        plot_limits_e = plot_limits[0] 
        plot_limits_r = plot_limits[1]
    else:
        plot_limits_e = None 
        plot_limits_r = None
    
    
    emin_annid = []  # список с минимальной энергией для разных сетей
    ravg_annid = []  # список с усредненным по разным hOmega радиусами для разных сетей
    
    
    ids_e = preds_energy['annid'].unique()  # номера нейросетей, которые остались
    ids_r = preds_radius['annid'].unique()  # номера нейросетей, которые остались
    
    # print(len(ids))
    for i in ids_e:  # цикл по сеткам
        Emin = preds_energy['Eabs'][preds_energy['annid'] == i][preds_energy['Nmax'] == n_energy].min()        
        emin_annid.append([Emin, i])
        
    for i in ids_r:  # цикл по сеткам        
        avgRMS = preds_radius['RMS'][preds_radius['annid'] == i][preds_radius['Nmax'] == n_radius].mean()
        ravg_annid.append([avgRMS, i])


    emin_annid = pd.DataFrame(emin_annid, columns=['E', 'annid']) # делаем датафрейм из списка
    ravg_annid = pd.DataFrame(ravg_annid, columns=['RMS', 'annid'])
       
    
    
    #************************************************
    # расчет что-то типа наиболее вероятного значения
    
    # считатем для энергии
    #################################################
    lower_bound = emin_annid['E'].min()
    upper_bound = emin_annid['E'].max()

    nsteps = 100 #число шагов для разбиения
    step = (upper_bound - lower_bound) / nsteps


    cwa = [] # c/w ratio, window, candidate for avg # cw_ratio = count / window
    
    for i in np.arange(lower_bound, upper_bound, step): # цикл по "кандидатам"
        for j in range(1, 2 * nsteps): # цикл по "окну"    
            window = step * j # пусть окно будет пропорционально шагу # просто так, может быть любым
            subset = emin_annid[emin_annid["E"] > i - window/2][emin_annid["E"] <= i + window/2] # выделение значений, входящих в окно
            
            counted = subset['E'].count()
            cwa.append([counted/window, window, i, counted])
            # if counted >= minimal_count:
            #     cwa.append([counted/window, window, i, counted])
            # else:
            #     cwa.append([0, window, i, 0])


    cwa = pd.DataFrame(cwa, columns = ['cw_ratio', 'window', 'avg', 'counted'])    
    #print(cwa)
    
    # ищем максимум
    # cwa['cw_ratio'].max()
    # ind = cwa['cw_ratio'].idxmax()
    # res = cwa.loc[ind]
       
    
    # newmu = res.avg
    # s = 0
       
    # # и отклонение
    # for i in range(len(emin_rms_annid)):
    #     #print(i)
    #     s += (emin_rms_annid['E'][i] - newmu) ** 2
    
    # s = math.sqrt(s / (len(emin_rms_annid) - 1))
    
    
    # но решено пойти другим путем
    # пусть приемлемое количество сетей, такое, что в окно умещаются предсказания 68% сетей
    # (как в стандартном распределение в стандартное отклонение умещается 68%)
    acc = math.ceil(0.6827 * len(preds_energy['annid'].unique())) 


    cwaacc = cwa.loc[cwa['counted'] > acc] # те, где наcчитано больше приемлемого количества
    cwaaccs = cwaacc.sort_values(by = ['cw_ratio'], ascending = False) # сортировка по убыванию
    
    #cwaaccs.head(5)
    
    # результаты
    mostfreq_E = float(cwaaccs.head(1)['avg']) # "наиболее вероятное значение"
    razbros_E  = float(cwaaccs.head(1)['window']) / 2 # "ст. отклонение"
    
    
    plt.xlabel('candidate value')
    plt.ylabel('count-to-window ratio')
    plt.plot(cwaacc['avg'], cwaacc['cw_ratio'])    
    plt.title("E", fontsize=30)
    plt.savefig(pics_path_e / plot_title / (plot_title + '_avg_cw_ratio_cwaacc' + '.' + plot_format), format = plot_format)
    #plt.show()
    plt.close()
    
    
    from matplotlib.ticker import LinearLocator
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_trisurf(cwa['window'], cwa['avg'], cwa['cw_ratio'])
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.set_axis_off()
    plt.savefig(pics_path_e / plot_title / (plot_title + '_window_avg_cw_ratio' + '.' + 'jpg'), format = 'jpg') # jpg пушо svg много весит
    #plt.show()
    plt.close()
    
    plt.xlabel('candidate value')
    plt.ylabel('count-to-window ratio')
    plt.plot(cwa['avg'], cwa['cw_ratio'])
    plt.title("E", fontsize=30)
    plt.savefig(pics_path_e / plot_title / (plot_title + '_avg_cw_ratio_cwa.' + plot_format), format = plot_format)        
    #plt.show()
    plt.close()
    
    
    ##################################################
    
    # считатем для радиуса (лень было нормально переделывать, поэтому тупо скопировал)
    #################################################
    lower_bound = ravg_annid['RMS'].min()
    upper_bound = ravg_annid['RMS'].max()

    nsteps = 100 #число шагов для разбиения
    step = (upper_bound - lower_bound) / nsteps


    cwa = [] # c/w ratio, window, candidate for avg # cw_ratio = count / window
    
    for i in np.arange(lower_bound, upper_bound, step): # цикл по "кандидатам"
        for j in range(1, 2 * nsteps): # цикл по "окну"    
            window = step * j # пусть окно будет пропорционально шагу # просто так, может быть любым
            subset = ravg_annid[ravg_annid["RMS"] > i - window/2][ravg_annid["RMS"] <= i + window/2]
            
            counted = subset['RMS'].count()
            cwa.append([counted/window, window, i, counted])
            # if counted >= minimal_count:
            #     cwa.append([counted/window, window, i, counted])
            # else:
            #     cwa.append([0, window, i, 0])


    cwa = pd.DataFrame(cwa, columns = ['cw_ratio', 'window', 'avg', 'counted'])    
    #print(cwa)
    
    # ищем максимум
    # cwa['cw_ratio'].max()
    # ind = cwa['cw_ratio'].idxmax()
    # res = cwa.loc[ind]
       
    
    # newmu = res.avg
    # s = 0
       
    # # и отклонение
    # for i in range(len(emin_rms_annid)):
    #     #print(i)
    #     s += (emin_rms_annid['E'][i] - newmu) ** 2
    
    # s = math.sqrt(s / (len(emin_rms_annid) - 1))
    
    
    # но решено пойти другим путем
    # пусть приемлемое количество сетей, такое, что в окно умещается предсказания 68% сетей
    # (как в стандартном распределение в стандартное отклонение уменщается 68%)
    acc = math.ceil(0.6827 * len(preds_radius['annid'].unique())) 


    cwaacc = cwa.loc[cwa['counted'] > acc] # те, где начитано больше приемлемого количества
    cwaaccs = cwaacc.sort_values(by = ['cw_ratio'], ascending = False) # сортировка по убыванию
    
    #cwaaccs.head(5)
    
    # результаты
    mostfreq_RMS = float(cwaaccs.head(1)['avg']) # "наиболее вероятное значение"
    razbros_RMS  = float(cwaaccs.head(1)['window']) / 2 # "ст. отклонение"
    
    
    plt.xlabel('candidate value')
    plt.ylabel('count-to-window ratio')
    plt.plot(cwaacc['avg'], cwaacc['cw_ratio'])    
    plt.title("RMS", fontsize=30) 
    plt.savefig(pics_path_r / plot_title / ('avg_cw_ratio_cwaacc.' + plot_format), format = plot_format)
    #plt.show()
    plt.close()
    
    
    from matplotlib.ticker import LinearLocator
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_trisurf(cwa['window'], cwa['avg'], cwa['cw_ratio'])
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.set_axis_off()
    plt.savefig(pics_path_r / plot_title / (plot_title + '_window_avg_cw_ratio.'+ plot_format), format = plot_format)
    #plt.show()    
    plt.close()
    
    plt.xlabel('candidate value')
    plt.ylabel('count-to-window ratio')
    plt.plot(cwa['avg'], cwa['cw_ratio'])
    plt.title("RMS", fontsize=30)  
    plt.savefig(pics_path_r / plot_title / (plot_title + '_avg_cw_ratio_cwa.' + plot_format), format = plot_format)        
    #plt.show()
    plt.close()
    #################################################
    #*************************************************
    
    
    
    
    avg_std_df = avg_and_std(preds_energy, preds_radius, n_energy, n_radius)
    
    # =====================================================
    # средние значения и всякие другие
    # avgE =         float(avg_and_std(preds_energy, preds_radius, n_energy, n_radius).avgE.astype(float))
    # stdE =         float(avg_and_std(preds_energy, preds_radius, n_energy, n_radius).stdE.astype(float))

    # medE =         float(avg_and_std(preds_energy, preds_radius, n_energy, n_radius).medE.astype(float))
    # midspreadE =   float(avg_and_std(preds_energy, preds_radius, n_energy, n_radius).midspreadE.astype(float))  # Q3-Q1
    
    # #посчитанные как бы наиболее вероятное значение и соотв. отклонение
    # mu_tildeE =    mostfreq_E # переименование тут, чтобы было в одном месте
    # sigma_tildeE = razbros_E

    # avgRMS =       float(avg_and_std(preds_energy, preds_radius, n_energy, n_radius).avgRMS.astype(float))
    # stdRMS =       float(avg_and_std(preds_energy, preds_radius, n_energy, n_radius).stdRMS.astype(float))
    
    # mu_tildeRMS =    mostfreq_RMS 
    # sigma_tildeRMS = razbros_RMS

    # medRMS =       float(avg_and_std(preds_energy, preds_radius, n_energy, n_radius).medRMS.astype(float))
    # midspreadRMS = float(avg_and_std(preds_energy, preds_radius, n_energy, n_radius).midspreadRMS.astype(float))  # Q3-Q1
    
    avgE =         float(avg_std_df.avgE)
    stdE =         float(avg_std_df.stdE)

    medE =         float(avg_std_df.medE)
    midspreadE =   float(avg_std_df.midspreadE)  # Q3-Q1

    EQ1 = float(avg_std_df.EQ1)
    EQ3 = float(avg_std_df.EQ3)
    
    #посчитанные как бы наиболее вероятное значение и соотв. отклонение
    mu_tildeE =    mostfreq_E # переименование тут, чтобы было в одном месте
    sigma_tildeE = razbros_E



    avgRMS =       float(avg_std_df.avgRMS)
    stdRMS =       float(avg_std_df.stdRMS)
    
    mu_tildeRMS =    mostfreq_RMS 
    sigma_tildeRMS = razbros_RMS

    medRMS =       float(avg_std_df.medRMS)
    midspreadRMS = float(avg_std_df.midspreadRMS)  # Q3-Q1

    RMSQ1 = float(avg_std_df.RMSQ1)
    RMSQ3 = float(avg_std_df.RMSQ3)


    # ===================================================

    # энергия
    # plt.figure('histE', figsize=(20.0, 28.0))
    fig, ax = plt.subplots(figsize=(20.0, 28.0))    

    # для определения ширины столбца (бина)
    if plot_limits is None:
        min_e = emin_annid['E'].min() 
        max_e = emin_annid['E'].max()
        hist_range = max_e - min_e
        if axis_bins is None:
            bin_width = hist_range / bins
        else:
            bin_width = hist_range / axis_bins
    else:
        if axis_bins is None:
            bin_width = (plot_limits_e[1] - plot_limits_e[0]) / bins
        else:
            bin_width = (plot_limits_e[1] - plot_limits_e[0]) / axis_bins


    plt.xlabel('min(E)')
    plt.ylabel('num of NNs')
    # plt.rc('axes', titlesize=30)  # fontsize of the axes title
    # plt.rc('axes', labelsize=30)  # fontsize of the x and y labels
    # plt.rc('xtick', labelsize=30)  # fontsize of the tick labels
    # plt.rc('ytick', labelsize=30)
    plt.rc('axes'  ,  titlesize = axes_titlesize)   # fontsize of the axes title
    plt.rc('axes'  ,  labelsize = axes_labelsize)   # fontsize of the x and y labels
    plt.rc('xtick' ,  labelsize = xtick_labelsize)  # fontsize of the tick labels
    plt.rc('ytick' ,  labelsize = ytick_labelsize)

    # Eplot_title = plot_title + f", energy value distribution , $E = {format(avgE, '.3f')} \pm {format(stdE, '.3f')}$"
    Eplot_title = plot_title + ", energy value distribution"

    textstr = '\n'.join((
        r'$\mu=%.3f$' % (avgE,),
        r'$\sigma=%.3f$' % (stdE,),
        r'$median = %.3f$' % (medE,),
        f'Q1 = {EQ1: .3f}',
        f'Q3 = {EQ3: .3f}',
        r'$midspread = %.3f$' % (midspreadE,),
        r'$\tilde{\mu} = %.3f$' % (mu_tildeE,),
        r'$\tilde{\sigma} = %.3f$' % (sigma_tildeE,),
        f'bin width = {round(bin_width, 3)}'))

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='tomato', alpha=0.5)

    # place a text box in upper left in axes coords
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=30,
            verticalalignment='top', bbox=props)


    #plt.axvline(x=mu_tildeE, color='red', linestyle='-', linewidth=4)  # "cреднее" значение предсказаний
    #ax.axvspan(mu_tildeE - sigma_tildeE, mu_tildeE + sigma_tildeE, facecolor='tomato', alpha=0.1)  # закрашивание "разброса" величины

    # среднее значение и отклонение на гистограмме в виде закрашенной области
    plt.axvline(x=avgE, color='red', linestyle='-', linewidth=4)  # cреднее значение предсказаний
    ax.axvspan(avgE - stdE, avgE + stdE, facecolor='tomato', alpha=0.1)  # закрашивание разброса величины

    #"разброс" величины
    plt.axvline(x=mu_tildeE,                color='firebrick', linestyle='-', linewidth=4)  # "cреднее" значение предсказаний
    plt.axvline(x=mu_tildeE - sigma_tildeE, color='firebrick', linestyle='--', linewidth=2, alpha = 0.5)
    plt.axvline(x=mu_tildeE + sigma_tildeE, color='firebrick', linestyle='--', linewidth=2, alpha = 0.5)

    # Q1, Q3 и медиана
    plt.axvline(x = EQ1, color = 'magenta', linestyle='--', linewidth = 2, alpha = 0.5)
    plt.axvline(x = EQ3, color = 'magenta', linestyle='--', linewidth = 2, alpha = 0.5)
    plt.axvline(x = medE, color = 'magenta', linestyle='-', linewidth = 4, alpha = 0.5)

    
    plt.title(Eplot_title, fontsize=30)
    plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
    plt.xticks(rotation='vertical')
    ax.xaxis.set_tick_params(width=4, length=10)
     
    
    if plot_limits is not None:
        ax.set_xlim(plot_limits_e) 

    
    if axis_bins is not None:        
        ax.xaxis.set_major_locator(plt.MaxNLocator('auto', prune = None, min_n_ticks = axis_bins))
        if plot_limits is not None:
            count_, bins_, bars_ = ax.hist(emin_annid.E, bins=axis_bins, edgecolor='black', color='tomato', range = plot_limits_e, density= True) # используется другое кол-во bins
        else:
            count_, bins_, bars_ = ax.hist(emin_annid.E, bins=axis_bins, edgecolor='black', color='tomato', density= True)
    else:
        count_, bins_, bars_ = ax.hist(emin_annid.E, bins=bins, edgecolor='black', color='tomato', density= True)

    #------------------------------------------------------------
    # гауссова кривулинка сверху
    x = np.linspace(emin_annid['E'].min(), emin_annid['E'].max(), 1000)
    y =       pdf(x, stdE, avgE) # нормировка на количество сетей, а не на 1 
    y_tilde = pdf(x, sigma_tildeE, mu_tildeE) #bin_width * len(ids_e) * 
    #print('x = ', x)
    #print('y = ', y)
    print(len(ids_e))
    line_1 = plt.plot(x, y,       color = 'black',     linewidth = 4)
    line_2 = plt.plot(x, y_tilde, color = 'firebrick', linewidth = 4)
    if plot_limits is not None:
        plt.xlim(plot_limits_e)

    # для проверки
    # xmin, xmax = plt.xlim()
    # x = np.linspace(xmin, xmax, 1000)
    # p = bin_width * len(ids_e) * norm.pdf(x, avgE, stdE)
    # plt.plot(x, p, 'k', linewidth=2)
    #------------------------------------------------------------
    plt.savefig(pics_path_e / plot_title / ("Гистограмма (норм) - " + Eplot_title + f'_bins={axis_bins}_' + str(plot_limits_e) + '.' + plot_format), format = plot_format)
    t = [b.remove() for b in bars_] # убирает гистограмму
    line_1 = line_1.pop(0) # убирает гауссоиду
    line_1.remove() 
    line_2 = line_2.pop(0)
    line_2.remove()  

  
    ax.relim()      # make sure all the data fits
    ax.autoscale()  # auto-scale
    if plot_limits is not None:
        ax.set_xlim(plot_limits_e)

    if axis_bins is not None:        
        ax.xaxis.set_major_locator(plt.MaxNLocator('auto', prune = None, min_n_ticks = axis_bins))
        if plot_limits is not None:
            count_, bins_, bars_ = ax.hist(emin_annid.E, bins=axis_bins, edgecolor='black', color='tomato', range = plot_limits_e) # используется другое кол-во bins
        else:
            count_, bins_, bars_ = ax.hist(emin_annid.E, bins=axis_bins, edgecolor='black', color='tomato')
    else:
        count_, bins_, bars_ = ax.hist(emin_annid.E, bins=bins, edgecolor='black', color='tomato')
    
    plt.savefig(pics_path_e / plot_title / ("Гистограмма - " + Eplot_title + f'_bins={axis_bins}_' + str(plot_limits_e) + '.' + plot_format), format = plot_format)
    #plt.show()

    

    
    #plt.show()
    plt.close()
    # ===============================
    
    # радиус
    # plt.figure('histR', figsize=(20.0, 28.0))
    fig, ax = plt.subplots(figsize=(20.0, 28.0))
    
    # для определения ширины столбца (бина)
    if plot_limits is None:
        min_ravg = ravg_annid['RMS'].min() 
        max_ravg = ravg_annid['RMS'].max()
        hist_range = max_ravg - min_ravg
        if axis_bins is None:
            bin_width = hist_range / bins
        else:
            bin_width = hist_range / axis_bins
    else:
        if axis_bins is None:
            bin_width = (plot_limits_r[1] - plot_limits_r[0]) / bins
        else:
            bin_width = (plot_limits_r[1] - plot_limits_r[0]) / axis_bins

    plt.xlabel('avg(RMS)')
    plt.ylabel('num of NNs')
    # plt.rc('axes', titlesize=30)  # fontsize of the axes title
    # plt.rc('axes', labelsize=30)  # fontsize of the x and y labels
    # plt.rc('xtick', labelsize=30)  # fontsize of the tick labels
    # plt.rc('ytick', labelsize=30)

    # Rplot_title = plot_title + f", radius value distribution , $RMS = {format(avgRMS, '.3f')} \pm {format(stdRMS, '.3f')}$"
    Rplot_title = plot_title + ", radius value distribution"

    textstr = '\n'.join((
        r'$\mu=%.3f$' % (avgRMS,),
        r'$\sigma=%.3f$' % (stdRMS,),
        r'$median = %.3f$' % (medRMS,),
        f'Q1 = {RMSQ1: .3f}',
        f'Q3 = {RMSQ3: .3f}',
        r'$midspread = %.3f$' % (midspreadRMS,),
        r'$\tilde{\mu} = %.3f$' % (mu_tildeRMS,),
        r'$\tilde{\sigma} = %.3f$' % (sigma_tildeRMS,),
        f'bin width = {round(bin_width, 3)}'))

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='gold', alpha=0.5)

    # place a text box in upper left in axes coords
    plt.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=30,
             verticalalignment='top', bbox=props)

    #plt.axvline(x=mu_tildeRMS, color='orange', linestyle='-', linewidth=4)  # "cреднее" значение предсказаний
    #ax.axvspan(mu_tildeRMS - sigma_tildeRMS, mu_tildeRMS + sigma_tildeRMS, facecolor='gold', alpha=0.1)  # закрашивание "разброса" величины
    
    # среднее значение и отклонение на гистограмме в виде закрашенной области
    plt.axvline(x=avgRMS, color='orange', linestyle='-', linewidth=4)  # cреднее значение предсказаний
    ax.axvspan(avgRMS - stdRMS, avgRMS + stdRMS, facecolor='gold', alpha=0.1)  # закрашивание разброса величины

    #"разброс" величины
    plt.axvline(x=mu_tildeRMS,                  color='darkgoldenrod', linestyle='-',  linewidth=4)  # "cреднее" значение предсказаний
    plt.axvline(x=mu_tildeRMS - sigma_tildeRMS, color='darkgoldenrod', linestyle='--', linewidth=2, alpha = 0.5)
    plt.axvline(x=mu_tildeRMS + sigma_tildeRMS, color='darkgoldenrod', linestyle='--', linewidth=2, alpha = 0.5)

    # Q1, Q3 и медиана
    plt.axvline(x = RMSQ1, color = 'magenta', linestyle='--', linewidth = 2, alpha = 0.5)
    plt.axvline(x = RMSQ3, color = 'magenta', linestyle='--', linewidth = 2, alpha = 0.5)
    plt.axvline(x = medRMS, color = 'magenta', linestyle='-', linewidth = 4, alpha = 0.5)

    plt.title(Rplot_title, fontsize=30)
    plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
    plt.xticks(rotation='vertical')
    ax.xaxis.set_tick_params(width=4, length=10)    
    
    if plot_limits is not None:
        ax.set_xlim(plot_limits_r)           
        
    if axis_bins is not None:        
        ax.xaxis.set_major_locator(plt.MaxNLocator(bins, prune = None, min_n_ticks = axis_bins))
        if plot_limits is not None:
            count_, bins_, bars_ = ax.hist(ravg_annid.RMS, bins=axis_bins, edgecolor='black', color='gold', range = plot_limits_r, density= True) # используется другое bins
        else: 
            count_, bins_, bars_ = ax.hist(ravg_annid.RMS, bins=axis_bins, edgecolor='black', color='gold', density= True)
    else:
        count_, bins_, bars_ = ax.hist(ravg_annid.RMS, bins=bins, edgecolor='black', color='gold', density= True)
    #------------------------------------------------------------
    # гауссова кривулинка сверху
    x = np.linspace(ravg_annid['RMS'].min(), ravg_annid['RMS'].max(), 1000)
    y =       pdf(x, stdRMS, avgRMS) # bin_width * len(ids_r) нормировка на количество сетей, а не на 1
    y_tilde = pdf(x, sigma_tildeRMS, mu_tildeRMS) 
    line_1 = plt.plot(x, y,       color = 'black',         linewidth = 4)
    line_2 = plt.plot(x, y_tilde, color = 'darkgoldenrod', linewidth = 4)
    if plot_limits is not None:
        plt.xlim(plot_limits_r)

    # для проверки
    # xmin, xmax = plt.xlim()
    # x = np.linspace(xmin, xmax, 1000)
    # p = bin_width * len(ids_r) * norm.pdf(x, avgRMS, stdRMS)
    # plt.plot(x, p, 'k', linewidth=2)
    #------------------------------------------------------------     
    
    plt.savefig(pics_path_r / plot_title / ("Гистограмма (норм) - " + Rplot_title + f'_bins={axis_bins}_' + str(plot_limits_r) + '.' + plot_format), format = plot_format)  
    t = [b.remove() for b in bars_] # убирает гистограмму
    line_1 = line_1.pop(0) # убирает гауссоиду
    line_1.remove() 
    line_2 = line_2.pop(0)
    line_2.remove() 

    ax.relim()      # make sure all the data fits
    ax.autoscale()  # auto-scale
    if plot_limits is not None:
        ax.set_xlim(plot_limits_r)

    if axis_bins is not None:        
        ax.xaxis.set_major_locator(plt.MaxNLocator(bins, prune = None, min_n_ticks = axis_bins))
        if plot_limits is not None:
            count_, bins_, bars_ = ax.hist(ravg_annid.RMS, bins=axis_bins, edgecolor='black', color='gold', range = plot_limits_r, density= True) # используется другое bins
        else: 
            count_, bins_, bars_ = ax.hist(ravg_annid.RMS, bins=axis_bins, edgecolor='black', color='gold', density= True)
    else:
        count_, bins_, bars_ = ax.hist(ravg_annid.RMS, bins=bins, edgecolor='black', color='gold', density= True) 
    plt.savefig(pics_path_r / plot_title / ("Гистограмма - " + Rplot_title + f'_bins={axis_bins}_' + str(plot_limits_r) + '.' + plot_format), format = plot_format)
    plt.close()
    
    # возвращаются "средние" значения и "ст. отклонения"
    return mu_tildeE, sigma_tildeE, mu_tildeRMS, sigma_tildeRMS, avgE, stdE, avgRMS, stdRMS 

def loss_histograms(data, label, path, pictures_path, bins, title, reverse_transform_args = None, figure_name_addition = ''): # гистограммы функции потерь на последней эпохе обучения, возвращается медианный loss
    # label - название столбца в датафрейме
    print('loss_histograms: begin')
    
       
    plot_title = title + ' ' + label + ' distribution'

    #avg = data[label].mean()  # среднее
    #std = data[label].std()  # стандартное отклонение
    med = data[label].median()  # медианное значение

    Q1 = data[label].quantile(q=0.25, interpolation='linear')  # первый (нижний) квартиль
    Q3 = data[label].quantile(q=0.75, interpolation='linear')  # третий (верхний) квартиль
    print(f'Q1 = {Q1}')
    print(f'Q3 = {Q3}')

    midspread = Q3 - Q1

    #--------------------------------------------------------------    
    loss_for_selection      = data[label].quantile(q = LOSS_SELECTION_QUANTILE, interpolation='linear') # определение величины loss, на основании которой будут отбираться сетки
    print(f'loss_histograms: loss_for_selection = {loss_for_selection}')
    data_selected           = data[data[label] < loss_for_selection] # отобранные данные
    annids_selected_by_loss = list(data_selected['annid'].unique())
    #--------------------------------------------------------------


    plt.figure('histL', figsize=(20.0, 28.0))
    plt.hist(data[label], bins=bins, edgecolor='black', color='royalblue')

    plt.xlabel('loss')
    plt.ylabel('num of NNs')

    # plt.rc('axes',  titlesize=30)  # fontsize of the axes title
    # plt.rc('axes',  labelsize=30)  # fontsize of the x and y labels
    # plt.rc('xtick', labelsize=30)  # fontsize of the tick labels
    # plt.rc('ytick', labelsize=30)

    # plot_title = plot_title + ', ' + r'$RMS = ' + str(format(avgRMS, '.3f')) + '\pm' + \
    #              str(format(stdRMS, '.3f')) + "$; " +'сетей: ' + str(number_of_nns) 

    plot_title = f'{plot_title}, median loss $ = {med:.3e}$, midspread $ = {midspread:.3e}$' \
                 f'\n loss(q={LOSS_SELECTION_QUANTILE}) = {loss_for_selection:.3e}, {len(data["annid"].unique())} ANNs'

    plt.title(plot_title, fontsize=30)

    # plt.xlim([0, 0.002])

    plt.axvline(x = med, color='black', linestyle='-')  # медианное значение
    plt.axvline(x = loss_for_selection, color='black', linestyle='dashed')  # значение loss для отбора: выбираются сетки со значением меньше данного

    plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    #plt.show()
    plt.savefig(os.path.join(pictures_path, (label + '_distribution' + figure_name_addition + '.' + plot_format)), format = plot_format)
    plt.close()

    # если даны доп. аргументы то делается обратное преобразование, и строится дополнительный график лосса, но "неотмасштабированных" значений
    if reverse_transform_args is not None:
        print('loss_histograms: rescaling begin')        
        (scale_ymax, scale_ymin, scale_max, scale_min) = reverse_transform_args
        PP_summary.write(str(reverse_transform_args))
        #print(reverse_transform_args)
        
        # обратное приоебразование, примерная ошибка
        data['sq_root_of_loss'] = np.sqrt(data[label])
        data['loss_rescaled'] = data['sq_root_of_loss']
        data['loss_rescaled'] *= (scale_ymax - scale_ymin) / (scale_max - scale_min)
        
        
        plot_title = title + ' ' + label + ' distribution (rescaled)'# + f'{reverse_transform_args}'

        #avg = data['loss_rescaled'].mean()  # среднее
        #std = data['loss_rescaled'].std()  # стандартное отклонение
        med = data['loss_rescaled'].median()  # медианное значение

        Q1 = data['loss_rescaled'].quantile(q=0.25, interpolation='linear')  # первый (нижний) квартиль
        Q3 = data['loss_rescaled'].quantile(q=0.75, interpolation='linear')  # третий (верхний) квартиль
        print(f'Q1 = {Q1}')
        print(f'Q3 = {Q3}')

        midspread = Q3 - Q1

        #--------------------------------------------------------------    
        loss_for_selection_rescaled      = data['loss_rescaled'].quantile(q = LOSS_SELECTION_QUANTILE, interpolation='linear') # определение величины loss, на основании которой будут отбираться сетки
        print(f'loss_histograms (rescaled): loss_for_selection_rescaled = {loss_for_selection_rescaled}')
        #data_selected           = data[data[label] < loss_for_selection] # отобранные данные
        #annids_selected_by_loss = list(data_selected['annid'].unique())
        #--------------------------------------------------------------

        plt.figure('histL', figsize=(20.0, 28.0))
        plt.hist(data['loss_rescaled'], bins=bins, edgecolor='black', color='cornflowerblue')

        plt.xlabel('loss')
        plt.ylabel('num of NNs')

        # plt.rc('axes',  titlesize=30)  # fontsize of the axes title
        # plt.rc('axes',  labelsize=30)  # fontsize of the x and y labels
        # plt.rc('xtick', labelsize=30)  # fontsize of the tick labels
        # plt.rc('ytick', labelsize=30)

        # plot_title = plot_title + ', ' + r'$RMS = ' + str(format(avgRMS, '.3f')) + '\pm' + \
        #              str(format(stdRMS, '.3f')) + "$; " +'сетей: ' + str(number_of_nns) 

        
        plot_title = f'{plot_title}, median loss $ = {med:.3e}$, midspread $ = {midspread:.3e}$' \
                 f'\n loss(q={LOSS_SELECTION_QUANTILE}) = {loss_for_selection_rescaled:.3e}, {len(data["annid"].unique())} ANNs'

        plt.title(plot_title, fontsize=30)

        # plt.xlim([0, 0.002])

        plt.axvline(x=med, color='black', linestyle='-')  # медианное значение
        plt.axvline(x = loss_for_selection_rescaled, color='black', linestyle='dashed')  # значение loss для отбора: выбираются сетки со значением меньше данного


        plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
        #plt.show()
        plt.savefig(os.path.join(pictures_path, (label + '_distribution_(rescaled)' + figure_name_addition + '.' + plot_format)), format = plot_format)
        plt.close()
        print('loss_histograms: rescaling end')     
    print('loss_histograms: end')      
    
    return med, annids_selected_by_loss

def weight_histograms(data, path, pictures_path, bins, title):  # гистограммы весов для разных слоев
    # label - название столбца в датафрейме
    # возвращается список коэффициентов остроты гистограмм

    pathlib.Path(pictures_path).mkdir(parents=True, exist_ok=True) 

    layers = data['layer'].unique()   

    spiciness_coefs = [] # список коэффициентов остроты для гистограммы для каждого слоя 
    # гистограмма для каждого слоя
    
    for layer in layers:

        layer_weights = data[data['layer'] == layer] # выбираем веса для данного слоя

        plot_title = f'{title}, layer {layer} weight distribution'

        #avg = data[label].mean()  # среднее
        #std = data[label].std()  # стандартное отклонение
        med = layer_weights['weight'].median()  # медианное значение

        Q1 = layer_weights['weight'].quantile(q=0.25, interpolation='linear')  # первый (нижний) квартиль
        Q3 = layer_weights['weight'].quantile(q=0.75, interpolation='linear')  # третий (верхний) квартиль
        #print(f'Q1 = {Q1}')
        #print(f'Q3 = {Q3}')

        midspread = Q3 - Q1


        # вычисление 99% вариабельности # штрихпунктир
        q001 = layer_weights['weight'].quantile(q=0.01, interpolation='linear')
        q099 = layer_weights['weight'].quantile(q=0.99, interpolation='linear')

        # вычисление 95% вариабельности # пунктир
        q005 = layer_weights['weight'].quantile(q=0.05, interpolation='linear')
        q095 = layer_weights['weight'].quantile(q=0.95, interpolation='linear')
        
        # коэффициент остроты 'графика'
        # coef = во сколько раз большая часть графика шире "главной" части (которая midspread)
        spiciness_coef = (q099 - q001) / midspread # spikiness 

        plt.figure('histW')
        plt.xlabel('weight')
        plt.ylabel('count')
        
        plot_title = f"{plot_title}, median weight $ = {med: .4f}$, midspread $ = {midspread: .4f}$ \n spiciness coef = {spiciness_coef: .3f}"
        plt.title(plot_title)

        plt.axvline(x=med,  color='black', linestyle='-')  # медианное значение
        
        plt.axvline(x=Q1,   color='black', linestyle='dashed')
        plt.axvline(x=Q3,   color='black', linestyle='dashed')

        plt.axvline(x=q095, color='black', linestyle='dashdot')
        plt.axvline(x=q005, color='black', linestyle='dashdot')

        plt.axvline(x=q099, color='black', linestyle='dotted')
        plt.axvline(x=q001, color='black', linestyle='dotted')
        
        


        #plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
        plt.hist(layer_weights['weight'], bins=bins, edgecolor='black', color='#38C141') # зеленый цвет
        plt.savefig(os.path.join(pictures_path, (title + 'layer_' + str(layer) + '_weight_distribution.' + plot_format)), format = plot_format)
        
        # -1 to 1
        plt.clf()
        plt.title(plot_title)
        plt.axvline(x=med, color='black', linestyle='-')
        plt.hist(layer_weights['weight'], bins=bins, edgecolor='black', color='#38C141', range=(-1, 1))        
        plt.savefig(os.path.join(pictures_path, (title + 'layer_' + str(layer) + '_weight_distribution_-1to1.' + plot_format)), format = plot_format)
        plt.close()         

        spiciness_coefs.append(spiciness_coef)
            
    return spiciness_coefs

def E_R_plots_and_histograms(preds_energy, preds_radius, maxN_energy, maxN_radius, n_energy, n_radius, hist_bins, variational_min, plot_title, # разные графики
                             E_exact_value=None, RMS_exact_value=None, E_extrapolation_value=None, plot_limits=None, bins = None):
    # ===
    start = time.perf_counter()    
    # #----------------
    
    pathlib.Path(pics_path_e / plot_title).mkdir(parents=True, exist_ok=True) #создание папки с названием  внутри pics_path_ для группировки картинок
    pathlib.Path(pics_path_r / plot_title).mkdir(parents=True, exist_ok=True)
    
    if plot_limits is not None:
        plot_limits_e = plot_limits[0] 
        plot_limits_r = plot_limits[1]
    else:
        plot_limits_e = None
        plot_limits_r = None
    
    
    
    
    avgstd = avg_and_std(preds_energy, preds_radius, n_energy, n_radius)
    avgE = float(avgstd.avgE.astype(float))
    stdE = float(avgstd.stdE.astype(float))
    
    #avgRMS = float(avgstd.avgRMS.astype(float))
    #stdRMS = float(avgstd.stdRMS.astype(float))
    
    
    
    print('Plotting energy...')
    energyplot(preds_energy, preds_radius, maxN_energy, n_energy, n_radius, variational_min, plot_title, 
               exact_value = E_exact_value, extrapolation_value = E_extrapolation_value, plot_limits = plot_limits_e, bins = bins)    
    plt.savefig(pics_path_e / plot_title / ("Зависимость минимума энергии от Nmax_" + plot_title + f'_bins={bins}_' + str(plot_limits_e) + '.' + plot_format), format = plot_format) # Добавлены лимиты в имя файла
    #plt.show()
    plt.close()

    
    print('Plotting radius...')
    # построение графиков RMS(hOmega) для разных Nmax 
    # деление на 10 чтобы огруглялось до Nmax кратного 10    
    for i in range (maxN_radius+10, n_radius, PREDICTIONS_FOR_NMAX_STEP):
        #rmsplot(preds_energy, preds_radius, maxN_radius, n_energy, math.floor(i/10)*10, plot_title + ', $N_{max} = ' + str(math.floor(i/10)*10) +'$', exact_value = RMS_exact_value, plot_limits = plot_limits_r, bins = bins)
        rmsplot(preds_energy, preds_radius, maxN_radius, n_energy, i, plot_title + ', $N_{max} = ' + str(math.floor(i/10)*10) +'$', exact_value = RMS_exact_value, plot_limits = plot_limits_r, bins = bins)
        plt.savefig(pics_path_r / plot_title / (f"Зависимость RMS от hOmega - Nmax = {str(math.floor(i/10)*10)}_" + plot_title + f'_bins={bins}_' + str(plot_limits_r) + '.' + plot_format), format = plot_format)  
        #plt.show()
        plt.close()
    
    
    # последний Nmax
    rmsplot(preds_energy, preds_radius, maxN_radius, n_energy, n_radius, plot_title + ', $N_{max} = ' + str(n_radius) +'$', exact_value = RMS_exact_value, plot_limits = plot_limits_r, bins = bins)
    plt.savefig(pics_path_r / plot_title / (f"Зависимость RMS от hOmega - Nmax = {n_radius}_" + plot_title + f'_bins={bins}_' + str(plot_limits_r) + '.' + plot_format), format = plot_format)  
    #plt.show()
    plt.close()
    
    print('Plotting histograms...')

    mu_tildeE, sigma_tildeE, mu_tildeRMS, sigma_tildeRMS, mu_E, sigma_E, mu_RMS, sigma_RMS = histograms(preds_energy, preds_radius, n_energy, n_radius, plot_title, hist_bins, plot_limits = plot_limits, axis_bins = bins)
    
    # t1 = threading.Thread(target=energyplot,
    #                       args=(preds, maxN, n, variational_min, plot_title),
    #                       kwargs={"exact_value": E_exact_value, "extrapolation_value": E_extrapolation_value})

    # t2 = threading.Thread(target=rmsplot,
    #                       args=(preds, maxN, n, plot_title),
    #                       kwargs={"exact_value": RMS_exact_value})
    # t3 = threading.Thread(target=histograms,
    #                       args=(preds, n, plot_title, hist_bins))

    # t1.start()
    # t2.start()
    # t3.start()

    # t1.join()
    # plt.show()

    # t2.join()
    # plt.show()

    # t3.join()

    # ----------------
    finish = time.perf_counter()
    print(f'Finished in {round(finish - start, 2)} second(s)')
    
    # возвращаются "средние" значения и "ст. отклонения"
    return mu_tildeE, sigma_tildeE, mu_tildeRMS, sigma_tildeRMS, mu_E, sigma_E, mu_RMS, sigma_RMS

def plot_limits(avgE, stdE, avgRMS, stdRMS, bounds_coef, offset_coef): #возвращает интервал для построения графика
    #------------------------------------------------
    #определение границ графиков для одинакового масштаба построения
    # bounds_coef для регулирования границ графика
    # offset для смещения среднего в единицах ст. отклонения
    #avgstd = avg_and_std(preds_energy, preds_radius, n_energy, n_radius)
    # avgstd = avg_and_std(preds_e, preds_r, n_e, n_r)
    # bounds_coef = 2.0
    # offset_coef = 0.0
    
    # #middleE_digit = 0 
    # #stepE_digit   = 1
    
    # #middleRMS_digit = 0 
    # #stepRMS_digit   = 1
    
    # avgE = float(avgstd.avgE)
    # stdE = float(avgstd.stdE)       
            
    # avgRMS = float(avgstd.avgRMS)
    # stdRMS = float(avgstd.stdRMS)  
    
    stdE = stdE * bounds_coef
    stdRMS = stdRMS * bounds_coef
    
    avgE = avgE + offset_coef * stdE
    avgRMS = avgRMS + offset_coef * stdRMS
    
    
    
    # для округления
    # lg_avgE = round(math.log10(abs(avgE)))
    # lg_stdE = round(math.log10(stdE))
    
    # lg_avgRMS = round(math.log10(abs(avgRMS)))
    # lg_stdRMS = round(math.log10(stdRMS))
    
    # digits = []
    
    # # опредение количества знаков округления
    # for x in [lg_avgE, lg_stdE, lg_avgRMS, lg_stdRMS]:            
    #     if x > 1.0:
    #         digits.append(0)
    #     else:                
    #         digits.append(math.ceil(abs(x)))
    
    # middleE_digit = digits[0]
    # stepE_digit   = digits[1]
    
    # middleRMS_digit = digits[2] 
    # stepRMS_digit   = digits[3]
            
    
    
    # if E_exact is not None: # если известно точное значение, то в качестве середины берется оно
    #     middleE = round(E_exact, middleE_digit)            
    # if E_exact is None:
    #     middleE = round(avgE, middleE_digit)        
    # stepE = round(stdE,stepE_digit)
    
    # # то же самое для радиуса
    # if rms_exact is not None: # если известно точное значение, то в качестве середины берется оно
    #     middleRMS = round(rms_exact, middleRMS_digit)            
    # if rms_exact is None:
    #     middleRMS = round(avgRMS, middleRMS_digit)        
    # stepRMS = round(stdRMS, stepRMS_digit)  
           
    # offset_e = round()
           
    
    # #bins_e = int(stepE * 10**stepE_digit) # количество отрезков на оси
    
    # #bins_r = int(stepRMS * 10**stepRMS_digit)
    
    # round_e = max(middleE_digit, stepE_digit)
    # round_r = max(middleRMS_digit, stepRMS_digit)
    
    
    ##выше делается странно - округляется до разных знаков, а по идее надо округлять всё до знака ст. откл
    
    # для округления
    lg_stdE   = math.log10(stdE)  
    lg_stdRMS = math.log10(stdRMS)
    
    # опредение количества знаков округления              
    if lg_stdE > 1.0:
        round_e = 0
    else:                
        round_e = math.ceil(abs(lg_stdE))
        
    if lg_stdRMS > 1.0:
        round_r = 0
    else:                
        round_r = math.ceil(abs(lg_stdRMS))
        
    middleE = avgE     + offset_coef * stdE
    middleRMS = avgRMS + offset_coef * stdRMS
    
    stepE = stdE
    stepRMS = stdRMS
    
    plot_limits_e = [round(middleE   - stepE,   round_e), round(middleE   + stepE,   round_e)]
    plot_limits_r = [round(middleRMS - stepRMS, round_r), round(middleRMS + stepRMS, round_r)]
    
    return plot_limits_e, plot_limits_r  

def plot_predictions_forNmax(predictions, dataframe, reference_df, whole_df, Nmax, path, energy_or_radius, plot_title): # предсказания для конкретного Nmax
    
    # whole_df - датафрейм для сравнения предсказаний с данными, которые были в исходном датафрейме (со всеми Nmax)    
    # предсказания строятся для модельного пространства равного Nmax
    # reference_df - датафрейм для выделенного модельного пространства
    # energy_or_radius - имя столбца: RMS или Eabs
    pathlib.Path(path / plot_title).mkdir(parents=True, exist_ok=True)
    #print(path)
    
    #print(whole_df)
       
    Nmax_from_df       = dataframe['Nmax'].unique()
    Nmax_from_ref_df   = list(reference_df['Nmax'].unique())
    Nmax_from_whole_df = list(whole_df['Nmax'].unique())
    
    Nmax_from_whole_df.sort()
    
    
    print(f'plot_predictions_forNmax: Nmax = {Nmax}') 
    #print(f'Nmax_from_df = {Nmax_from_df}')   
    #print(f'Nmax_from_ref_df = {Nmax_from_ref_df}')  
    #print(f'Nmax_from_whole_df = {Nmax_from_whole_df}')

    #print(f'predictions = \n{predictions}')
    #print(f'df = \n{dataframe}')
    #print(f'reference_df = \n{reference_df}')
    #print(f'whole_df = \n{whole_df}')
    
    # ищем предыдущее и следующее модельные пр-во для Nmax [для которого строятся предсказания]
    if Nmax in Nmax_from_whole_df:
        Nmax_idx = Nmax_from_whole_df.index(Nmax)
        prev_idx = Nmax_idx - 1
        next_idx = Nmax_idx + 1
        
        if prev_idx >= 0: 
            prev_Nmax = Nmax_from_whole_df[prev_idx]
        else:
            prev_Nmax = None
            
        if next_idx < len(Nmax_from_whole_df):
            next_Nmax = Nmax_from_whole_df[next_idx]
        else:
            next_Nmax = None
            
    else:
        prev_Nmax = None
        next_Nmax = None
    
    
    #print(f'prev_Nmax = {prev_Nmax}')
    #print(f'next_Nmax = {next_Nmax}')
    

    for ann_id in predictions['annid'].unique():        
        forplot = predictions.loc[(predictions['annid'] == ann_id) & (predictions['Nmax'] == Nmax)]
        #print(f'forplot = \n{forplot}')        
        plt.plot(forplot['hOmega'], forplot[energy_or_radius])        
    
    # если заданный Nmax содержится в данных для обучения или в данных для метрики, то строится черная линия
    if Nmax in Nmax_from_df:        
        plt.plot(dataframe['hOmega'][dataframe['Nmax'] == Nmax], dataframe[energy_or_radius][dataframe['Nmax'] == Nmax], color = 'black', linewidth = 4)
        
    if Nmax in Nmax_from_ref_df:
        plt.plot(reference_df['hOmega'], reference_df[energy_or_radius], color = 'black', linewidth = 4)
    
    # следующий Nmax
    if next_Nmax is not None:
        plt.plot(whole_df['hOmega'][whole_df['Nmax'] == next_Nmax], whole_df[energy_or_radius][whole_df['Nmax'] == next_Nmax], color = 'grey', linewidth = 4, linestyle = 'dashed')
    # предыдущий Nmax
    if prev_Nmax is not None:
        plt.plot(whole_df['hOmega'][whole_df['Nmax'] == prev_Nmax], whole_df[energy_or_radius][whole_df['Nmax'] == prev_Nmax], color = 'grey', linewidth = 4, linestyle = 'dashed')
    
    #текущий Nmax, если он не содержится ни в тренировчном датасете, ни в датасете для отбора по Deviation, но содержится в исходном датасете
    if (Nmax not in list(Nmax_from_df)) and (Nmax not in Nmax_from_ref_df) and (Nmax in list(Nmax_from_whole_df)):
        plt.plot(whole_df['hOmega'][whole_df['Nmax'] == Nmax], whole_df[energy_or_radius][whole_df['Nmax'] == Nmax], color = 'dimgrey', linewidth = 4, linestyle = 'dashed')
        
        
        
    
    
    plt.ylabel(f'${energy_or_radius}$')
    plt.xlabel('$\hbar \Omega$')
    
    number_of_nns = len(predictions['annid'].unique())
    
    plt.title(f'Предсказания {energy_or_radius} для Nmax = {Nmax}, сетей: {number_of_nns}')
    if Nmax in Nmax_from_ref_df:
        plt.title(f'Предсказания {energy_or_radius} для выделенного модельного пр-ва: {Nmax}, сетей: {number_of_nns}')
    
    plt.savefig(path / plot_title / (f'Предсказания {energy_or_radius} для Nmax = {Nmax}, {plot_title}.' + plot_format), format = plot_format)
    #plt.show()
    plt.close()
                                 
def iterative_outlier_filtering_of_predictions(dataframe_e, dataframe_r, method: str = 'boxplot_rule', max_iterations: int = 1, stopping_coef: float = 1e-5):
    '''
    итеративная фильтрация выбросов из предсказаний методом "boxplot_rule" (default) или методом "n_sigma"

    метод boxplot_rule заключается в вычислении медианы и интерквартильного размаха: X is outlier if X < Q1 - 1.5*IQR or X > Q3 + 1.5*IQR
    IQR = Q3 - Q1 --- 75- и 25-процентили
    есть разные модификации этого метода

    метод n_sigma основан на вычислении серднего значения и среднеквадратичного отклонения: X is outlier if X < mu - n*sigma or X > mu + n*sigma

    передаются два df (датафрейма) потому что для радиуса и для энергии слегка разная обработка:
    радиус усредняется по hOmega, а для энергии берется минимум

    !!! коэффициент для определения границ должен содержаться в конфигурационном файле
    k - максимальное число итераций
    stopping_coef - критерий остановки: если относительная разница между предыдущим
    значением и последующим меньше заданного, то итерации прекращаются; 
    в качестве значения для мониторинга берется location - среднее значение или медиана для описываемых методов
    '''    
    
    if method != 'boxplot_rule' and method != 'n_sigma':
        print(f'method = {method}')
        raise Exception('iterative_outlier_filtering_of_predictions: method argument is incorrect')
    
    # определения для variance и location в зависимости от метода
    #------------------------------------------------------------
    def calculate_location(df):
        val   = df.columns.values.tolist()[0] # названия столбцов
        #annid = df.columns.values.tolist()[1]

        if method == 'boxplot_rule':
            return df[val].median()
        
        if method == 'n_sigma':
            return df[val].mean()


    def calculate_variance(df):
        val   = df.columns.values.tolist()[0] # названия столбцов
        #annid = df.columns.values.tolist()[1]

        if method == 'boxplot_rule':
            return df[val].quantile(q = 0.75) - df[val].quantile(q = 0.25) # вычисление IQR
        
        if method == 'n_sigma':
            return df[val].std()
        
    #------------------------------------------------------------

    df_e = dataframe_e.copy()
    df_r = dataframe_r.copy()
    
    # конвертация столбца Nmax в int на всякий случай: при загрузке из файла тип float64 
    df_e = df_e.astype({"Nmax": int}, errors='raise')
    df_r = df_r.astype({"Nmax": int}, errors='raise')
    
    maxN_e = df_e.Nmax.max()
    maxN_r = df_r.Nmax.max()
    
    emin_annid = []  # список с минимальной энергией для разных сетей
    ravg_annid = []  # список с усредненным по разным hOmega радиусами для разных сетей

    ids_e = preds_e['annid'].unique()  # номера нейросетей
    ids_r = preds_r['annid'].unique()  

    for i in ids_e:  # цикл по сеткам
        Emin   = df_e['Eabs'][df_e['annid'] == i][df_e['Nmax'] == maxN_e].min()        
        emin_annid.append([Emin, i])
        
    for i in ids_r:  # цикл по сеткам        
        avgRMS = df_r['RMS'][df_r['annid']  == i][df_r['Nmax'] == maxN_r].mean()
        ravg_annid.append([avgRMS, i])


    emin_annid = pd.DataFrame(emin_annid, columns=['E', 'annid']) # делаем датафрейм из списка
    ravg_annid = pd.DataFrame(ravg_annid, columns=['RMS', 'annid'])    
    
    
    def iterative_filtering(dataframe):        
        # предполагается следующая структура датафрейма:
        # col1 = values, col2 = quasi_index
        
        
        def filter_outliers(dataframe):
            # предполагается следующая структура датафрейма:
            # col1 = values, col2 = quasi_index      
            # location и variance - вычисляемые метрики           
           
            
            df = dataframe.copy()   
            
            val   = df.columns.values.tolist()[0] # названия столбцов
            #annid = df.columns.values.tolist()[1]

            # определение границ, значения вне которых считаются выбросами
            location = calculate_location(df)
            variance = calculate_variance(df)

            if method == 'boxplot_rule':   
                # ошибка  
                # left_bound  = location - BOXPLOT_RULE_COEF * variance
                # right_bound = location + BOXPLOT_RULE_COEF * variance
                left_bound  = df[val].quantile(q = 0.25) - BOXPLOT_RULE_COEF * variance
                right_bound = df[val].quantile(q = 0.75) + BOXPLOT_RULE_COEF * variance
            
            if method == 'n_sigma':           
                left_bound  = location - N_SIGMA_SELECTION * variance
                right_bound = location + N_SIGMA_SELECTION * variance
            
            df.drop(df[df[val] > right_bound].index, inplace=True)
            df.drop(df[df[val] <  left_bound].index, inplace=True)
            
            return df
        
        df = dataframe.copy()
        
        val   = df.columns.values.tolist()[0] # названия столбцов
        #annid = df.columns.values.tolist()[1]
        
        # sigma = df[val].std()
        # mean  = df[val].mean()               
                       
        location = calculate_location(df)
        variance = calculate_variance(df)
        
        print(f'initial value: {location} \u00B1 {variance}')
        
        #mean_old = mean
        #mean_new = None
        
        variance_old = variance
        variance_new = None
        
        for i in range(max_iterations):
            print('iterative_outlier_filtering_of_predictions: begin...')
            print(f'iterative_filtering: iteraion # {i+1} of maximum {max_iterations}')
            df = filter_outliers(df)

                         
            location_new = calculate_location(df)
            variance_new = calculate_variance(df)            
            
            print(f'new value: {location_new} \u00B1 {variance_new}')
            
            variance_rel_change = abs(variance_old - variance_new) / variance_old
            #mean_rel_change  = abs(mean_old  - mean_new)  / mean_old
            
            print(f'variance relative change: {variance_rel_change: .2e}')
            #print(f'mean  relative change: { mean_rel_change: .2e}')
            
            if  variance_rel_change < stopping_coef:# or  mean_rel_change < stopping_coef:
                break
            else:
                #mean_old  = mean_new
                variance_old = variance_new  
            print('iterative_outlier_filtering_of_predictions: end...')              
        
        return df   
    
    emin_annid_selected = iterative_filtering(emin_annid)
    ravg_annid_selected = iterative_filtering(ravg_annid)
    
    # plt.title(f'E = {emin_annid_selected.E.mean()} \u00B1 {emin_annid_selected.E.std()}')
    # plt.hist(emin_annid_selected.E, bins = 80, range = (-32.2, -31.8))
    # plt.show()

    # plt.title(f'RMS = {ravg_annid_selected.RMS.mean()} \u00B1 {ravg_annid_selected.RMS.std()}')
    # plt.hist(ravg_annid_selected.RMS, bins = 80, range = (2.3, 2.7))
    # plt.show()
    
    annids_selected_e = emin_annid_selected.annid.unique() # номера отобранных сетей
    annids_selected_r = ravg_annid_selected.annid.unique()
    
    df_e_selected = df_e[df_e['annid'].isin(annids_selected_e)] # оставляем только отобранные сетки
    df_r_selected = df_r[df_r['annid'].isin(annids_selected_r)]    

    return df_e_selected, df_r_selected

def Nmax_convergence(data, pictures_path, bins, title, eps_nmax, eps_ho):
    # data - датафрейм предсказаний энергии  
    # title - добавка к имени файла и к названию
    # возвращает медианный Nmax, на котором сетки сходятся и список сеток, которые сходятся около Nmax_conv      

    annids = sorted(data['annid'].unique())
    Nmaxes = sorted(data['Nmax' ].unique())

    maxNmax = max(Nmaxes)
    Nmax_unconv = maxNmax + 50 # Nmax, который будет на графике считаться за "несошедшийся"

    step = Nmaxes[1] - Nmaxes[0]
    #range(min_nmax_step, max_nmax_step, min_nmax_step) #для цикла по шагам


    # вычисляется для каждой сетки и для каждого Nmax разница между
    # 1) макс. и мин. энергией
    # 2) разница между минимумами энергии между следующим (на шаг) и предыдущим Nmax
    # то есть если разница "-", то энергия стала меньше

    differences_df = pd.DataFrame(columns=['Nmax', 'dif_E_min_max', 'dif_E_Nmax_prev_next', 'annid'])
    dif_E_min_max             = [] # 1)
    dif_E_Nmax_prev_next      = [] # 2)
    Nmaxes_for_differences_df = []
    annids_for_differences_df = []

    for i in annids:
        selected_by_id = data[data['annid'] == i]
        for Nmax in range(Nmaxes[0], Nmaxes[-1] - step, step):
            Nmaxes_for_differences_df.append(Nmax)
            annids_for_differences_df.append(i)
            
            selected_by_id_nmax = selected_by_id[selected_by_id['Nmax'] == Nmax]
            E_min = selected_by_id_nmax['Eabs'].min()
            E_max = selected_by_id_nmax['Eabs'].max()
            
            dif_E_min_max.append(E_min - E_max)

            selected_by_id_plus_step_nmax = selected_by_id[selected_by_id['Nmax'] == (Nmax + step)]
            E_min_next = selected_by_id_plus_step_nmax['Eabs'].min()
            
            E_min_prev = E_min
            dif_E_Nmax_prev_next.append(E_min_next - E_min_prev)


    differences_df['Nmax']                     = Nmaxes_for_differences_df
    differences_df['dif_E_min_max']            = dif_E_min_max
    differences_df['abs_dif_E_min_max']        = differences_df['dif_E_min_max'].abs()
    differences_df['dif_E_Nmax_prev_next']     = dif_E_Nmax_prev_next
    differences_df['abs_dif_E_Nmax_prev_next'] = differences_df['dif_E_Nmax_prev_next'].abs()
    differences_df['annid']                    = annids_for_differences_df



    # сразу убираются строки, не удовлетворяющие условиям
    # dropped_dif_df = differences_df[differences_df['abs_dif_E_min_max']        <= EPS_HO]
    # print(f"residual by EPS_HO condition {len(dropped_dif_df['annid'].unique())}")
    # dropped_dif_df = dropped_dif_df[dropped_dif_df['abs_dif_E_Nmax_prev_next'] <= EPS_NMAX * step]
    # print(f"residual by EPS_Nmax condition {len(dropped_dif_df['annid'].unique())}")

    # в другом порядке
    dropped_dif_df = differences_df[differences_df['abs_dif_E_Nmax_prev_next'] < eps_nmax * step]
    print(f"Nmax_convergence: residual by EPS_NMAX condition {len(dropped_dif_df['annid'].unique())}")
    dropped_dif_df = dropped_dif_df[dropped_dif_df['abs_dif_E_min_max']        < eps_ho]
    print(f"Nmax_convergence: residual by EPS_HO condition {len(dropped_dif_df['annid'].unique())}")
    
    # типа с проверкой вариационного принципа    
    # dropped_dif_df = differences_df[differences_df['dif_E_Nmax_prev_next'] < eps_nmax * step]
    # dropped_dif_df = dropped_dif_df[dropped_dif_df['dif_E_Nmax_prev_next'] > 0]
    # print(f"Nmax_convergence: residual by EPS_NMAX condition {len(dropped_dif_df['annid'].unique())}")
    # dropped_dif_df = dropped_dif_df[dropped_dif_df['abs_dif_E_min_max']        < eps_ho]
    # print(f"Nmax_convergence: residual by EPS_HO condition {len(dropped_dif_df['annid'].unique())}")

    # определение Nmax, при котором сетки сходятся
    Nmaxes_conv = []
    annids_conv = sorted(dropped_dif_df['annid'].unique())
    step_col    = []
    for i in annids_conv:
        Nmaxes_conv.append(dropped_dif_df[dropped_dif_df['annid'] == i]['Nmax'].min()) # минмальный Nmax, при котором есть сходимость
        step_col.append(step)


    conv_df = pd.DataFrame(columns = ['step', 'Nmax_conv', 'annid'])
    conv_df['step']      = step_col
    conv_df['Nmax_conv'] = Nmaxes_conv
    conv_df['annid']     = annids_conv

    #print(conv_df)

    # определение медианного Nmax, при котором сетки сходятся
    median_Nmax_conv = conv_df['Nmax_conv'].median()  # медианное значение

    # ИЗНАЧАЛЬНО БРАЛСЯ IQR, ТО ЕСТЬ Q1 И Q3 - ПЕРВЫЙ И ТРЕТИЙ КВАРТИЛИ, НО ПОТОМ БЫЛО РЕШЕНО ВЗЯТЬ С НУЛЯ
    # ПОЭТОМУ В ПОДПИСЯХ МОЖЕТ БЫТЬ ФИГНЯ
    Q1 = conv_df['Nmax_conv'].quantile(q = CONV_LEFT_QUANTILE, interpolation='nearest')  # первый (нижний) квартиль
    Q3 = conv_df['Nmax_conv'].quantile(q = CONV_RIGHT_QUANTILE, interpolation='nearest')  # третий (верхний) квартиль
    midspread = Q3 - Q1 # в измененном виде ни фига это не midspread 
    

    print(f'Nmax_convergence: median_Nmax_conv = {median_Nmax_conv}')
    #print(f'Q1 = {Q1}')
    #print(f'Q3 = {Q3}')

    # определение той части сеток, где в большая часть их сходится
    # выбираются Nmax между Q1 и Q3 
    aboveQ1 = set(conv_df['annid'][conv_df['Nmax_conv'] >= Q1].unique())
    belowQ3 = set(conv_df['annid'][conv_df['Nmax_conv'] <= Q3].unique())
    
    #print(aboveQ1)
    #print(belowQ3)

    average_Nmaxes_conv_annids = aboveQ1.intersection(belowQ3)
    # доля сошедшихся сеток, которые лежат в между Q1 и Q3 по отношению ко всем сошедшимся сеткам 
    Q1Q3_annids_ratio = len(average_Nmaxes_conv_annids) / (len(annids_conv))

    #print(f'average_Nmaxes_conv = {average_Nmaxes_conv_annids}')
    #print(len(average_Nmaxes_conv_annids))


    # заполнение несошедшихся сеток каким-то большим Nmax
    unconv_df = pd.DataFrame(columns = ['step', 'Nmax_conv', 'annid'])
    annids_unconv = list(set(annids) - set(annids_conv))
    Nmaxes_unconv = [Nmax_unconv] *  len(annids_unconv)
    step_unconv   = [step] * len(annids_unconv)
    unconv_df['step']      = step_unconv
    unconv_df['Nmax_conv'] = Nmaxes_unconv
    unconv_df['annid']     = annids_unconv

    # сшивка датафреймов
    conv_unconv_df = pd.concat([conv_df, unconv_df])


    plt.hist(conv_unconv_df['Nmax_conv'], bins = bins, edgecolor='black', color='#7F6087') # лиловый цвет

    plt.axvline(x=median_Nmax_conv, color='black', linestyle='-')  # медианное значение
    plt.axvline(x=Q1, color='black', linestyle='dashdot')  # медианное значение
    plt.axvline(x=Q3, color='black', linestyle='dashdot')  # медианное значение

    plt.xlabel('Nmax')
    plt.ylabel('num of nns')

    conv_ratio = len(annids_conv) / len(annids)
    plot_title = f"{title}Nmax, при котором достигнута сходимость \n EPS_HO = {eps_ho}, EPS_NMAX = {eps_nmax} \n convergence ratio = {conv_ratio: .2f} \n midspread = {midspread: d} \n median Nmax_conv = {median_Nmax_conv} \n Nmax_left = {Q1}, Nmax_right = {Q3}"
    plt.title(plot_title)


    #plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    plt.savefig(pictures_path / (title + 'Nmax_conv' + str(eps_ho) + '_' + str(eps_nmax) + '.'+ plot_format), format = plot_format)
    #print(Nmaxes_conv)
    #print(dropped_dif_df)
    #print(dropped_dif_df['annid'].unique())
    plt.close()

    print(len(conv_unconv_df['annid'].unique()))
    if len(conv_unconv_df['annid'].unique()) != len(annids):
        raise Exception('Nmax convergence: number of anns is not equal!')

    return median_Nmax_conv, average_Nmaxes_conv_annids

def scale_fn(x):     
    return 1.0 / x ** CLR_SCALE_FN_POWER 

def plot_predictions_for_hOmega(predictions, used_data_df, whole_df, pic_path, energy_or_radius, plot_title): # предсказания для конкретного Nmax
    '''
    Предсказания энергии или радиуса в зависимости от Nmax при данном hO для всех сеток \n
    used_data_df - датафрейм и использованными (для обучения и для валидации, например) данными, он используется для определения наименьшего Nmax, в частности \n
    whole_df используется для сравнения полученных предсказаний (черная линия на графике)    
    '''

    pathlib.Path(pic_path / plot_title).mkdir(parents=True, exist_ok=True)

    Nmaxes_from_whole_df     = sorted(list(whole_df[    'Nmax' ].unique()))
    hOmegas_from_whole_df    = sorted(list(whole_df[   'hOmega'].unique())) 
    Nmaxes_from_predictions  = sorted(list(predictions['Nmax'  ].unique()))    
    hOmegas_from_predictions = sorted(list(predictions['hOmega'].unique())) 
    number_of_nns = len(predictions['annid'].unique())  
     
    print(f'plot_predictions_for_hOmega: Nmaxes_from_whole_df = {Nmaxes_from_whole_df}')
    print(f'plot_predictions_for_hOmega: Nmaxes_from_predictions = {Nmaxes_from_predictions}')
    print(f'plot_predictions_for_hOmega: hOmegas_from_whole_df = {hOmegas_from_whole_df}')
    print(f'plot_predictions_for_hOmega: hOmegas_from_predictions = {hOmegas_from_predictions}')

    
    hOmega_step = hOmegas_from_predictions[1] - hOmegas_from_predictions[0] # шаг по hO
    print(f'plot_predictions_for_hOmega: hOmega_step = {hOmega_step}')

    #определение нового шага по hO с учетом минимального шага   
    if hOmega_step < PREDICTIONS_FOR_HOMEGA_MINIMAL_STEP:
        i = 0
        hO_step_new = hOmega_step
        while hO_step_new < PREDICTIONS_FOR_HOMEGA_MINIMAL_STEP:
            i += 1
            hO_step_new = hOmegas_from_predictions[1 + i] - hOmegas_from_predictions[0]
        del i
    else:
        hO_step_new = hOmega_step
    
    # переопределение набора hO с учетом нового шага
    # чтобы при случае включить правую границу
    hOmegas_for_plotting = list(np.arange(hOmegas_from_predictions[0], hOmegas_from_predictions[-1] + hO_step_new, hO_step_new))
    hOmegas_for_plotting = set(hOmegas_for_plotting).intersection(set(hOmegas_from_predictions))
    hOmegas_for_plotting = sorted(list(hOmegas_for_plotting))

    print(f'plot_predictions_for_hOmega: hO_step_new = {hO_step_new}')
    print(f'plot_predictions_for_hOmega: hOmegas_for_plotting = {hOmegas_for_plotting}')


    # границы по Nmax (условно, для красоты)
    Nmax_left  = used_data_df['Nmax'].min()
    Nmax_right = whole_df['Nmax'].max()

    Nmax_ticks = list(range(Nmax_left, max(Nmaxes_from_predictions), 20))
    Nmax_ticks.append(max(Nmaxes_from_predictions))
    
    for hOmega_for_plotting in hOmegas_for_plotting:
        print(f'plot_predictions_for_hOmega: plotting for hO = {hOmega_for_plotting}...')
        #fig, ax = plt.subplots()
        y_limit_max = predictions[energy_or_radius][(predictions['hOmega'] == hOmega_for_plotting) & (predictions['Nmax'] >= Nmax_left)].max() # для красоты
        if y_limit_max >= 0.0:
            y_limit_max = 1.1 * y_limit_max 
        else:
            y_limit_max = 0.9 * y_limit_max 
    
        y_limit_min = predictions[energy_or_radius][(predictions['hOmega'] == hOmega_for_plotting) & (predictions['Nmax'] >= Nmax_left)].min() # для красоты 
        if y_limit_min >= 0.0:
            y_limit_min = 0.9 * y_limit_min
        else:
            y_limit_min = 1.1 * y_limit_min 

        for ann_id in predictions['annid'].unique():        
            forplot = predictions.loc[(predictions['annid'] == ann_id) & (predictions['hOmega'] == hOmega_for_plotting)]
            #print(f'forplot = \n{forplot}')        
            plt.plot(forplot['Nmax'], forplot[energy_or_radius])        
    
        # строится черная линия по исходным данным
        if hOmega_for_plotting in hOmegas_from_whole_df:
            plt.plot(whole_df['Nmax'][whole_df['hOmega'] == hOmega_for_plotting], whole_df[energy_or_radius][whole_df['hOmega'] == hOmega_for_plotting], color = 'black', linewidth = 4)
            
        plt.ylabel(f'${energy_or_radius}$')
        plt.xlabel('$N_{max}$')  
        
        plt.title(f'Предсказания {energy_or_radius} для $\hbar\Omega$ = {hOmega_for_plotting}, сетей: {number_of_nns}')   
        
        #plt.xticks(ticks = Nmaxes_from_predictions)        
        plt.xticks(ticks = Nmax_ticks)
        plt.xlim(xmin = Nmax_left)
        plt.ylim(ymax = y_limit_max, ymin = y_limit_min)                
        #ax.axes.relim(visible_only=False)
        plt.xticks(rotation='vertical')
        plt.savefig(pic_path / plot_title / (f'Предсказания {energy_or_radius} для hO = {hOmega_for_plotting}.' + plot_format), format = plot_format)
        plt.xlim(xmax = Nmax_right)        
        plt.xticks(ticks = Nmaxes_from_whole_df)
        plt.savefig(pic_path / plot_title / (f'Предсказания {energy_or_radius} для hO = {hOmega_for_plotting}, Nmax_right = {Nmax_right}.' + plot_format), format = plot_format)
        #plt.show()
        plt.close()

def variational_principle_check(preds, input_data):
    '''
    # проверяется вариационный принцип
    # используются предсказания и используемые исходные данные
    # в качестве отправной точки используется самая нижняя кривая в используемых входных данных

    # preds - датафрейм с предсказаниями
    # input_data - данные, использованные для обучения    


    # шаг по hOmega подразумевается равномерным как в данных, так и в предсказаниях

    # сначала определяется набор hOmega, при которых проверяется вариационный принцип
    
    # константы:
    # VP_hO_left - левая граница по hOmega при проверке вариационного принципа
    # VP_hO_right - правая граница
    # VP_hO_minimal_step - минимальный шаг по hO: если шаг в предсказаниях меньше минимального, то используется заданный минимальный шаг, а не тот, который в предсказаниях
    # VP_epsilon - допустимая разница при сравнении энергии при соседних Nmax при данном hOmega
    '''
    print(f'variational_principle_check: BEGIN....')
    ids = sorted(preds.annid.unique())

    if VP_HO_RIGHT < VP_HO_LEFT:
        raise Exception('variational_principle_check: VP_hO_right < VP_hO_left!')

    # определение нижайшей кривой E(hO), которая будет использоваться для проверки вариационного принципа
    max_Nmax = input_data['Nmax'].max()

    match VP_CHECK_MODE:
        case 1: #первое сравнение - сравнение предсказаний следующего Nmax с исходными данными
            bottom = input_data[['Nmax', 'hOmega', 'Eabs']][input_data['Nmax'] == max_Nmax]
            PP_summary.write('\nvariational_principle_check: mode 1')             
        case 2: # первое сравнение - сравнение предсказаний следующего Nmax с предсказаниями
            # выбираются предсказания первой сетки, это надо лишь для определения hO, в которых будет проверяться в. п.
            # дальше bottom переопределится, так что будут выбраны предсказания нужной сетки
            bottom = preds[['Nmax', 'hOmega', 'Eabs']].loc[(preds['annid'] == ids[0]) & (preds['Nmax'] == max_Nmax)]
            PP_summary.write('\nvariational_principle_check: mode 2')              
        case _:
            PP_summary.write('\nvariational_principle_check: mode incorrect')
            raise Exception("variational_principle_check: VP_CHECK_MODE is not correct")
    
    
    print(f'max_Nmax = {max_Nmax}')
    print(f'bottom: \n{bottom}')


    hOmegas_preds      = sorted(preds['hOmega'].unique())      
    hOmegas_input_data = sorted(input_data['hOmega'].unique())

    # нужные hOmega содержатся именно в нижней кривой
    hOmegas_bottom     = sorted(bottom['hOmega'].unique())

    hOmegas_common     = set(hOmegas_preds).intersection(set(hOmegas_bottom)) # общие hO для hOmegas_preds и hOmegas_bottom

    print(f'hOmegas_preds = {hOmegas_preds}, length = {len(hOmegas_preds)}')
    print(f'hOmegas_input_data = {hOmegas_input_data}, length = {len(hOmegas_input_data)}')
    print(f'hOmegas_bottom = {hOmegas_bottom}, length = {len(hOmegas_bottom)}')
    print(f'hOmegas_common = {hOmegas_common}, length = {len(hOmegas_common)}')

    preds = preds[preds['hOmega'].isin(hOmegas_input_data)] # на всякий случай выбираются только те hO, которые содержатся и в исходных данных, и в предсказаниях
    input_data = input_data[input_data['hOmega'].isin(hOmegas_preds)]

    #df_e_selected = df_e[df_e['annid'].isin(annids_selected_e)]
    
    hOmegas_common = sorted(list(hOmegas_common))
    hOmega_step = hOmegas_common[1] - hOmegas_common[0] # шаг по hO
    print(f'hOmega_step = {hOmega_step}')
    
    # определение нового шага по hO с учетом старого шага    
    if hOmega_step < VP_HO_MINIMAL_STEP:
        i = 0
        hO_step_new = hOmega_step
        while hO_step_new < VP_HO_MINIMAL_STEP:
            i += 1
            hO_step_new = hOmegas_common[1 + i] - hOmegas_common[0]
        del i
    else:
        hO_step_new = hOmega_step
    
    print(f'hO_step_new = {hO_step_new}')

    #hOmegas_common = sorted(list(hOmegas_common))
    # первый поиск левой и правой границ среди общих hO (hOmegas_common), т.е. левая граница - такое hO_left_new, что hO_left_new >= VP_hO_left
    # чтобы начать именно с левой границы при учете заданного шага    
    VP_hO_left_new = VP_HO_LEFT
    VP_hO_right_new = VP_HO_RIGHT

    for hO in hOmegas_common:
        if hO >= VP_HO_LEFT:
            VP_hO_left_new = hO
            break
    
    # если заданная граница больше правой границы уже имеющегося диапазона, то правая граница - граница диапазона
    # а если нет, то ищется среди hO ближайжее к заданному значение  
    if VP_HO_RIGHT >= hOmegas_common[-1]:
        VP_hO_right_new = hOmegas_common[-1]
    else:            
        for hO in reversed(hOmegas_common): # reversed - итерации с конца
            if hO <= VP_HO_RIGHT:
                VP_hO_right_new = hO
                break
    print(f'VP_hO_left_new (initial) = {VP_hO_left_new}')
    print(f'VP_hO_right_new (initial) = {VP_hO_right_new}')
    
    # выбираются hO больше определенной левой границы и меньше правой границы
    hOmegas_common = [hO for hO in hOmegas_common if hO >= VP_hO_left_new ]
    hOmegas_common = [hO for hO in hOmegas_common if hO <= VP_hO_right_new]
    print(f'hOmegas_common (final) = {hOmegas_common}')

    # переопределение интервала hO с учетом шага
    hOmegas_new = list(np.arange(hOmegas_common[0], hOmegas_common[-1] + hO_step_new, hO_step_new))
    print(f'numpy arange hOmegas_new = {hOmegas_new}')
    # выбор тех hO, которые содержались в hOmegas_new, ибо проблемы с границами могут быть
    hOmegas_new = set(hOmegas_new).intersection(set(hOmegas_common))
    hOmegas_new = sorted(list(hOmegas_new))
    print(f'hOmegas_new (final) = {hOmegas_new}')
    
    # окончательное определение граничных hO (для занесения в summary)
    VP_hO_left_new  = hOmegas_new[0] # по идее, левая начальная левая граница должна совпадать с конечной 
    VP_hO_right_new = hOmegas_new[-1]
    print(f'VP_hO_left_new (final) = {VP_hO_left_new}')
    print(f'VP_hO_right_new (final) = {VP_hO_right_new}')

    PP_summary.write(f'\nVP: VP_hO_left = {VP_HO_LEFT}')
    PP_summary.write(f'\nVP: VP_hO_right = {VP_HO_RIGHT}')
    PP_summary.write(f'\nVP: VP_hO_minimal_step = {VP_HO_MINIMAL_STEP}')
    PP_summary.write(f'\nVP: VP_step_new = {hO_step_new}')
    PP_summary.write(f'\nVP: VP_hO_left_new = {VP_hO_left_new}')
    PP_summary.write(f'\nVP: VP_hO_right_new = {VP_hO_right_new}')
    PP_summary.write(f'\nVP: hOmegas_new = {hOmegas_new}')

    # очистка данных от ненужных hO
    preds = preds[preds['hOmega'].isin(hOmegas_new)]
    input_data = input_data[input_data['hOmega'].isin(hOmegas_new)]
    bottom = bottom[bottom['hOmega'].isin(hOmegas_new)]
    #data_for_metric = data_for_metric[data_for_metric['hOmega'].isin(hOmegas_new)]

    print(f'preds (filtered): \n{preds}')
    print(f'input_data (filtered): \n{input_data}')
    print(f'bottom (filtered): \n{bottom}')
    
    #PP_summary.write('VP: input_data (filtered) = {input_data}')
    PP_summary.write(f'\nVP: bottom (filtered) = {bottom}')

    if len(bottom) == 0:
        raise Exception('bottom curve is empty!')

    #print(f'data_for_metric = {data_for_metric}')

    # # определение нижайшей кривой E(hO), которая будет использоваться для проверки вариационного принципа
    # if input_data['Nmax'].max() > Nmax_for_metric_e:
    #     max_Nmax = df_e['Nmax'].max()
    #     bottom = df_e[['hOmega', 'Eabs']][df_e['Nmax'] == max_Nmax] # нижняя кривая в имеющихся данных
    # else:
    #     max_Nmax = Nmax_for_metric_e
    #     bottom = df_for_metric_e[['hOmega', 'Eabs']]

    droplist = []  # список сетей для отбрасывания на основания вариац. принципа
    

    Nmaxes_preds = sorted(preds['Nmax'].unique())
    Nmax_step = Nmaxes_preds[1] - Nmaxes_preds[0] # шаг по Nmax в предсказаниях

    # if Nmax_step != NMAX_PREDICTIONS_STEP:
    #     raise Exception('Nmax step from the predictions is not equal to Nmax step from the config')    

    # собственно отбор по вар. пр.
    dropcounter = 0 # список, сеток, которые не прошли вариационный принцип
    var_princ_list = [] # T/F список для хранения информации о том, выполнен ли вариационный принцип или нет (при данном hO и данном Nmax) 
    var_princ_dif_list = [] # список для хранения разницы между энергиями при данном hO при соседних Nmax

    for i in ids:  # цикл по сеткам (для каждой сети смотрим предсказания) 
                    
        match VP_CHECK_MODE:
            case 1: #первое сравнение - сравнение предсказаний следующего Nmax с исходными данными
                prev = bottom['Eabs'].to_numpy().reshape(-1, 1)  # инициализация нижней кривой в имеющихся данных
            case 2: # первое сравнение - сравнение предсказаний следующего Nmax с предсказаниями
                # выбираются предсказания первой сетки, это надо лишь для определения hO, в которых будет проверяться в. п.
                # переопределение bottom, так что будут выбраны предсказания нужной сетки
                bottom = preds[['Nmax', 'hOmega', 'Eabs']].loc[(preds['annid'] == ids[i]) & (preds['Nmax'] == max_Nmax)]
                bottom = bottom[bottom['hOmega'].isin(hOmegas_new)]
                prev = bottom['Eabs'].to_numpy().reshape(-1, 1)          
            case _:
                raise Exception("variational_principle_check: VP_CHECK_MODE is not correct")

        #print(prev)        
        #print('i = ', i)        
        if i % 100 == 0:
            print(f"var princ: i = {i} of {len(ids)}")

        for N in range(max_Nmax + Nmax_step, n_variational_e + Nmax_step, Nmax_step):  # цикл по Nmax; max_Nmax - максимальное мод. пр-во в input_data
            #print("N=", N)
            # n+step шобы включить n
            # maxN + step пушо в данных отброшены Nmax, большие maxN

            # проверяем выполнение вариационного принципа:            
            # смотрим как лежит предсказанная энергия: выше или ниже последней кривой.to_numpy().reshape(-1,1)
            #print('prev = ', prev)
            nex = preds[['hOmega', 'Eabs']].loc[(preds['annid'] == i) & (preds['Nmax'] == N)]
            #print('nex from preds = ', nex) 

            nex = nex['Eabs'].to_numpy().reshape(-1, 1)  # оставляем только энергию
            #print('nex = ', nex)                          

            # способ сразу через сравнение (без информации о величине и без допустимой ошибки)
            #---------------------------------
            #compar = np.less_equal(nex, prev)  # element-wise comparison of the arrays
            #var_princ_list.extend(compar.tolist())
            #varprinc = np.prod(compar) # поэлементное произведение compar; слишком жирно, можно было сделать и проще
            #---------------------------------

            # способ через вычисление разницы
            #---------------------------------            
            nex_prev_difference = np.subtract(nex, prev) # разница
            #print(f'nex_prev dif = {nex_prev_difference}')
            var_princ_dif_list.extend(nex_prev_difference.tolist()) # разница копится для графика
            compar = nex_prev_difference <= VP_EPSILON
            var_princ_list.extend(compar.tolist())
            varprinc = np.prod(compar) # поэлементное произведение compar; слишком жирно, можно было сделать и проще
            #---------------------------------
                  
            
            if varprinc == 0:  # если вариац. принцип не выполнен, то сеть выбрасываем
                # print("N = ", N, "model №: ", i, "var principle is not fulfilled")
                droplist.append(i)
                # preds.drop(preds[preds.annid == i].index, inplace = True)
                # print(preds[preds.annid == i].index)
                # preds = preds[preds.annid != i]
                dropcounter += 1

            prev = nex  # потом будем сравнивать с последним с предсказанием
    
    droplist = set(droplist) # шоб быстрее 
    print(f'VP: dropcounter = {dropcounter}')
    print(f'VP: len(droplist) = {len(droplist)}, \n droplist = {droplist}')
    PP_summary.write(f'\nVP: len(droplist) = {len(droplist)}, \n droplist = {droplist}')   


    # костыль:
    # если отброшены были почти все сети, то не отбрасываем ничего, чтобы продолжить выполнение программы
    if len(ids) - len(droplist) < MIN_MODELS_e:
        PP_summary.write('вариационный принцип не выполнен почти для всех сетей, продолжение без отбора.....')
        droplist = []

    # для построения графиков, иллюстрирующих выполнение вариационного принципа
    var_princ_list = [val[0] for val in var_princ_list]# выше возвращается [[true], [true]...], поэтому [0]
    var_princ_dif_list = [val[0] for val in var_princ_dif_list]


    preds_for_VP_plot = preds.copy()
    Nmaxes_for_VP_plot = list(range(max_Nmax + Nmax_step, n_variational_e + Nmax_step, Nmax_step)) #Nmax, которые участвовали при проверке вариационного принципа

    preds_for_VP_plot = preds_for_VP_plot[preds_for_VP_plot['Nmax'].isin(Nmaxes_for_VP_plot)] # предсказания, которые участвовали при проверке в. п.
    preds_for_VP_plot['var_princ'] = var_princ_list
    preds_for_VP_plot['var_princ_dif'] = var_princ_dif_list
    #print(preds_for_VP_plot)
    #print(var_princ_list)


    
    preds_for_VP_plot_true  = preds_for_VP_plot[preds_for_VP_plot['var_princ'] == True]  # предсказания, для которых локально выполнен в. п.
    preds_for_VP_plot_false = preds_for_VP_plot[preds_for_VP_plot['var_princ'] == False] # предсказания, для которых локально НЕ выполнен в. п.

    # количество сеток прошедших в. п. и не прошедших в. п.
    num_of_anns_VP_positive = len(preds['annid'].unique()) - len(droplist)
    num_of_anns_VP_negative = len(droplist)

    # предсказания, для которых вариационный принцип выполнен (глобально, т. е. в целом)
    preds_for_VP_plot_ok     = preds_for_VP_plot[~preds_for_VP_plot['annid'].isin(droplist)]
    preds_for_VP_plot_not_ok = preds_for_VP_plot[ preds_for_VP_plot['annid'].isin(droplist)] 

    # построение разных графиков
    # bins_hO_true  = len(preds_for_VP_plot_true[ 'hOmega'].unique()) - 1
    # bins_hO_false = len(preds_for_VP_plot_false['hOmega'].unique()) - 1
    # bins_Nmax_true  = len(preds_for_VP_plot_true[ 'Nmax'].unique()) - 1
    # bins_Nmax_false = len(preds_for_VP_plot_false['Nmax'].unique()) - 1


    
    #------------------------------------------------------
    # гистограмма для разницы энергий при проверке вар. пр. 
    # границы интервала для гистограммы
    # там, где нужна только отриц. или полож часть, границы вычисляются из a,b: [-a, 0.1*b] или [0.1*a, b]
    a = -0.01 # левая
    b =  0.01 # правая
    energy_dif_bins = 100 # число разбиений для гистограммы

    plt.hist(preds_for_VP_plot['var_princ_dif'], edgecolor='black', bins = energy_dif_bins, color = 'mediumaquamarine')
    plt.ylabel('count')
    plt.xlabel('разница по энергии между соседними Nmax')
    plt.title(f'распределение разницы по энергии. \nКоличество точек: {len(preds_for_VP_plot)}, количество сетей: {len(preds_for_VP_plot["annid"].unique())}')    
    plt.savefig(pics_path_e / ('VP_check_dif_hist.' + plot_format), format = plot_format)
    plt.close()

    # гистограмма для разницы энергий (range)
    fig, ax = plt.subplots()
    ax.ticklabel_format(useMathText=True)   
    plt.hist(preds_for_VP_plot['var_princ_dif'], edgecolor='black', bins = energy_dif_bins, color = 'mediumaquamarine', range = (a, b))
    plt.ylabel('count')
    plt.xlabel('разница по энергии между соседними Nmax')
    plt.title(f'распределение разницы по энергии. \nКоличество точек: {len(preds_for_VP_plot)}, количество сетей: {len(preds_for_VP_plot["annid"].unique())}')    
    plt.savefig(pics_path_e / ('VP_check_dif_hist_range.' + plot_format), format = plot_format)
    plt.close()

    # гистограмма для разницы энергий log 
    plt.hist(preds_for_VP_plot['var_princ_dif'], edgecolor='black', bins = energy_dif_bins, color = 'mediumaquamarine', log = True)
    plt.ylabel('count')
    plt.xlabel('разница по энергии между соседними Nmax')
    plt.title(f'распределение разницы по энергии. \nКоличество точек: {len(preds_for_VP_plot)}, количество сетей: {len(preds_for_VP_plot["annid"].unique())}')    
    plt.savefig(pics_path_e / ('VP_check_dif_hist_log.' + plot_format), format = plot_format)
    plt.close()

    # гистограмма для разницы энергий log (range)
    plt.hist(preds_for_VP_plot['var_princ_dif'], edgecolor='black', bins = energy_dif_bins, color = 'mediumaquamarine', log = True, range = (a, b))
    plt.ylabel('count')
    plt.xlabel('разница по энергии между соседними Nmax')
    plt.title(f'распределение разницы по энергии. \nКоличество точек: {len(preds_for_VP_plot)}, количество сетей: {len(preds_for_VP_plot["annid"].unique())}')    
    plt.savefig(pics_path_e / ('VP_check_dif_hist_log_range.' + plot_format), format = plot_format)
    plt.close()

    # гистограмма для разницы энергий (range) T
    fig, ax = plt.subplots()
    ax.ticklabel_format(useMathText = True)   
    plt.hist(preds_for_VP_plot_true['var_princ_dif'], edgecolor='black', bins = energy_dif_bins, color = 'pink', range = (a, 0.1*b))
    plt.ylabel('count')
    plt.xlabel('разница по энергии между соседними Nmax')
    plt.title(f'распределение разницы по энергии: вариационный принцип локально выполнен. \nКоличество точек: {len(preds_for_VP_plot_true)}, количество сетей: {len(preds_for_VP_plot_true["annid"].unique())}')    
    plt.savefig(pics_path_e / ('VP_check_dif_hist_range_positive.' + plot_format), format = plot_format)
    plt.close()

    # гистограмма для разницы энергий (range) F
    fig, ax = plt.subplots()
    ax.ticklabel_format(useMathText = True)   
    plt.hist(preds_for_VP_plot_false['var_princ_dif'], edgecolor='black', bins = energy_dif_bins, color = 'powderblue', range = (0.1*a, b))
    plt.ylabel('count')
    plt.xlabel('разница по энергии между соседними Nmax')
    plt.title(f'распределение разницы по энергии: вариационный принцип локально не выполнен. \nКоличество точек: {len(preds_for_VP_plot_false)}, количество сетей: {len(preds_for_VP_plot_false["annid"].unique())}')    
    plt.savefig(pics_path_e / ('VP_check_dif_hist_range_negative.' + plot_format), format = plot_format)
    plt.close()   

    # гистограмма для разницы энергий (range) ok 
    fig, ax = plt.subplots()
    ax.ticklabel_format(useMathText = True) 
    plt.hist(preds_for_VP_plot_ok['var_princ_dif'], edgecolor='black', color = 'hotpink', bins = energy_dif_bins, range = (a, 0.1*b))
    plt.ylabel('count')
    plt.xlabel('разница по энергии между соседними Nmax')
    plt.title(f'распределение разницы по энергии: вариационный принцип выполнен. \nКоличество точек: {len(preds_for_VP_plot_ok)}, количество сетей: {len(preds_for_VP_plot_ok["annid"].unique())}')    
    plt.savefig(pics_path_e / ('VP_check_dif_hist_range_ok.' + plot_format), format = plot_format)
    plt.close()

    # гистограмма для разницы энергий (range) not ok
    fig, ax = plt.subplots()
    ax.ticklabel_format(useMathText = True)  
    plt.hist(preds_for_VP_plot_not_ok['var_princ_dif'], edgecolor='black', color = 'slateblue', bins = energy_dif_bins, range = (a, b))
    plt.ylabel('count')
    plt.xlabel('разница по энергии между соседними Nmax')
    plt.title(f'распределение разницы по энергии: вариационный принцип не выполнен. \nКоличество точек: {len(preds_for_VP_plot_not_ok)}, количество сетей: {len(preds_for_VP_plot_not_ok["annid"].unique())}')    
    plt.savefig(pics_path_e / ('VP_check_dif_hist_range_not_ok.'  + plot_format), format = plot_format)
    plt.close()
    #------------------------------------------------------    
    

    #======================================================
    # гистограмма (зависимость количества точек, где в. п. выполнен от hO) для hO T   
    #plt.hist(preds_for_VP_plot_true['hOmega'], edgecolor='black', bins =  bins_hO_true)
    # гистограмма выглядит плохо (последний столбец в 2 раза больше), ибо все интервалы имеют вид [a,b), а последний - [a,b]
    # поэтому строится график количества при данном hO
    x = sorted(preds_for_VP_plot_true['hOmega'].unique())
    y = []
    for hO in x:
        y.append(len(preds_for_VP_plot_true[preds_for_VP_plot_true['hOmega'] == hO]))       
    plt.plot(x, y, marker='.', linestyle = 'dashed', color = 'pink')     
    plt.ylabel('count')
    plt.xlabel('$\hbar \Omega$')
    plt.title(f'вариационный принцип выполнен при данном $\hbar \Omega$: количество. \nОбщее количество точек: {len(preds_for_VP_plot_true)}, количество сетей: {len(preds_for_VP_plot_true["annid"].unique())}')    
    plt.savefig(pics_path_e / ('VP_check_hOmega_hist_positive.' + plot_format), format = plot_format)
    plt.ylim(bottom = 0, top = max(y) * 1.1)
    plt.savefig(pics_path_e / ('VP_check_hOmega_hist_positive_top_bottom.' + plot_format), format = plot_format) 
    plt.close()

    # гистограмма для hO F        
    x = sorted(preds_for_VP_plot_false['hOmega'].unique())
    y = []
    for hO in x:
        y.append(len(preds_for_VP_plot_false[preds_for_VP_plot_false['hOmega'] == hO]))
    plt.plot(x, y, marker='.', linestyle = 'dashed', color = 'cadetblue') 
    plt.ylabel('count')
    plt.xlabel('$\hbar \Omega$')    
    plt.title(f'вариационный принцип не выполнен при данном $\hbar \Omega$: количество. \nОбщее количество точек: {len(preds_for_VP_plot_false)}, количество сетей: {len(preds_for_VP_plot_false["annid"].unique())}') 
    plt.savefig(pics_path_e / ('VP_check_hOmega_hist_negative.' + plot_format), format = plot_format)
    plt.ylim(bottom = 0, top = max(y) * 1.1)
    plt.savefig(pics_path_e / ('VP_check_hOmega_hist_negative_top_bottom.' + plot_format), format = plot_format)
    plt.close()
    #======================================================


    #******************************************************
    # гистограмма для Nmax T
    x = sorted(preds_for_VP_plot_true['Nmax'].unique())
    y = []
    for N in x:
        y.append(len(preds_for_VP_plot_true[preds_for_VP_plot_true['Nmax'] == N])) 
    plt.plot(x, y, marker='.', linestyle = 'dashed', color = 'palevioletred') 
    plt.ylabel('count')    
    plt.xlabel('$N_{max}$')
    plt.title('вариационный принцип выполнен при данном $N_{max}$: количество. ' + f'\nОбщее количество точек: {len(preds_for_VP_plot_true)}, количество сетей: {len(preds_for_VP_plot_true["annid"].unique())}')
    plt.savefig(pics_path_e / ('VP_check_Nmax_hist_positive.' + plot_format), format = plot_format)
    plt.ylim(bottom = 0, top = max(y) * 1.1) 
    plt.savefig(pics_path_e / ('VP_check_Nmax_hist_positive_top_bottom.' + plot_format), format = plot_format)
    plt.close()

    # гистограмма для Nmax F
    x = sorted(preds_for_VP_plot_false['Nmax'].unique())
    y = []
    for N in x:
        y.append(len(preds_for_VP_plot_false[preds_for_VP_plot_false['Nmax'] == N])) 
    plt.plot(x, y, marker='.', linestyle = 'dashed', color = 'lightblue')
    plt.ylabel('count')    
    plt.xlabel('$N_{max}$')
    plt.title('вариационный принцип не выполнен при данном $N_{max}$: количество. ' f'\nОбщее количество точек: {len(preds_for_VP_plot_false)}, количество сетей: {len(preds_for_VP_plot_false["annid"].unique())}') 
    plt.savefig(pics_path_e / ('VP_check_Nmax_hist_negative.' + plot_format), format = plot_format)
    plt.ylim(bottom = 0, top = max(y) * 1.1) 
    plt.savefig(pics_path_e / ('VP_check_Nmax_hist_negative_top_bottom.' + plot_format), format = plot_format)
    plt.close()
    #******************************************************


    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # точечный график для hO при разных annid T
    plt.scatter(preds_for_VP_plot_true['annid'], preds_for_VP_plot_true['hOmega'], s = 4)   
    plt.xlabel('annid')
    plt.ylabel('$\hbar \Omega$')
    plt.title(f'$\hbar \Omega$, где вариационный принцип выполнен. \n Общее количество точек: {len(preds_for_VP_plot_true)} количество сетей: {num_of_anns_VP_positive}')
    plt.savefig(pics_path_e / ('VP_check_hOmega_scatter_positive.' + plot_format), format = plot_format)
    plt.close()    
    
    # точечный график для hO при разных annid F
    plt.scatter(preds_for_VP_plot_false['annid'], preds_for_VP_plot_false['hOmega'], s = 4)    
    plt.xlabel('annid')
    plt.ylabel('$\hbar \Omega$')
    plt.title(f'$\hbar \Omega$, где вариационный принцип не выполнен \n Общее количество точек: {len(preds_for_VP_plot_false)} количество сетей: {num_of_anns_VP_negative}') 
    plt.savefig(pics_path_e / ('VP_check_hOmega_scatter_negative.' + plot_format), format = plot_format)
    plt.close()
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    

    #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # точечный график для Nmax при разных annid T
    plt.scatter(preds_for_VP_plot_true['annid'], preds_for_VP_plot_true['Nmax'], s = 4)
    plt.xlabel('annid')
    plt.ylabel('$N_{max}$')
    plt.title('$N_{max}$, где вариационный принцип выполнен \n количество сетей: ' + f'\nОбщее количество точек: {len(preds_for_VP_plot_true)}, количество сетей: {len(preds_for_VP_plot_true["annid"].unique())}')
    plt.savefig(pics_path_e / ('VP_check_Nmax_scatter_positive.' + plot_format), format = plot_format)
    plt.close()    
    
    # точечный график для Nmax при разных annid F
    plt.scatter(preds_for_VP_plot_false['annid'], preds_for_VP_plot_false['Nmax'], s = 4)    
    plt.xlabel('annid')
    plt.ylabel('$N_{max}$')
    plt.title('$N_{max}$, где вариационный принцип не выполнен \n количество сетей: '+ f'\nОбщее количество точек: {len(preds_for_VP_plot_false)}, количество сетей: {len(preds_for_VP_plot_false["annid"].unique())}') 
    plt.savefig(pics_path_e / ('VP_check_Nmax_scatter_negative.' + plot_format), format = plot_format)
    plt.close()
    #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  


    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # гистограмма строится некорректно (см выше)
    # двумерная гистограмма T
    fig = plt.figure()
    # для бинов
    bin_edges_hO   = []
    bin_edges_Nmax = []
    hOmegas_true = sorted(preds_for_VP_plot_true['hOmega'].unique())
    Nmaxes_true  = sorted(preds_for_VP_plot_true['Nmax'].unique())

    #bin_edges_hO.append(hOmegas_true[0] - hO_step_new)
    bin_edges_hO.extend(hOmegas_true)
    bin_edges_hO.append(hOmegas_true[-1] + hO_step_new)

    #bin_edges_Nmax.append(Nmaxes_true[0] - Nmax_step)
    bin_edges_Nmax.extend(Nmaxes_true)
    bin_edges_Nmax.append(Nmaxes_true[-1] + Nmax_step)
    
    print(f'bin_edges_hO = {bin_edges_hO}')
    print(f'bin_edges_Nmax = {bin_edges_Nmax}')

    plt.hist2d(preds_for_VP_plot_true['hOmega'], preds_for_VP_plot_true['Nmax'], bins=(bin_edges_hO, bin_edges_Nmax), cmap = 'plasma')
    #plt.imshow(a, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('распределение $\hbar \Omega$ и $N_{max}$, где вариационный принцип выполнен локально \n количество сетей: ' + f'{len(preds_for_VP_plot_true)}')
    plt.xlabel('$\hbar \Omega$')     
    plt.ylabel('$N_{max}$')
    plt.savefig(pics_path_e / ('VP_check_heatmap_positive.' + plot_format), format = plot_format)
    plt.close()

    # двумерная гистограмма F
    fig = plt.figure()
    plt.hist2d(preds_for_VP_plot_false['hOmega'], preds_for_VP_plot_false['Nmax'], bins=(bin_edges_hO, bin_edges_Nmax), cmap = 'plasma')
    plt.colorbar()
    plt.title('распределение $\hbar \Omega$ и $N_{max}$, где вариационный принцип не выполнен локально \n количество сетей: ' + f'{len(preds_for_VP_plot_false)}')
    plt.xlabel('$\hbar \Omega$')     
    plt.ylabel('$N_{max}$')
    plt.savefig(pics_path_e / ('VP_check_heatmap_negative.' + plot_format), format = plot_format)
    plt.close()
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # это НЕ номера сеток, прошедших вариационный принцип, это номера сеток, где вариационный принцип выполнен локально (может сбить с толку)
    VP_passed_ids_local = sorted(preds_for_VP_plot_true['annid'].unique())
    # print(f'VP_passed_ids = {VP_passed_ids}')
    # print(f'len(VP_passed_ids) = {len(VP_passed_ids)}')


    # номера сеток, прошедшних в. п.
    VP_passed_ids = sorted(preds_for_VP_plot_ok['annid'].unique())

    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # дополнительная проверка:
    # сравнивается последнее модельное пространствое, которое участвовало в проверке вариационного принципа (Nmax = Nmax_1)
    # с последним модельным пространством, которое есть в предсказаниях (Nmax = Nmax_2),
    # но сравнение не на всем диапазоне Nmax, а всего лишь этих двух мод. пр-в

    # предполагается, шо Nmax_2 ощутимо больше Nmax_1
    if VP_ADDITIONAL_CHECK == True:
        # выбор данных
        preds_max_variational_Nmax = preds.loc[preds['Nmax'] == n_variational_e] # наибольшее мод. пр-во, которое участвовало в проверке в. п.. 
        preds_max_variational_Nmax = preds_max_variational_Nmax[preds_max_variational_Nmax['hOmega'].isin(hOmegas_new)]

        max_Nmax_in_preds = preds['Nmax'].max()
    
        preds_max_Nmax = preds.loc[preds['Nmax'] == max_Nmax_in_preds] # предсказания для последнего мод. пр-ва
        preds_max_Nmax = preds_max_Nmax[preds_max_Nmax['hOmega'].isin(hOmegas_new)] # убираются ненужные hO
        
        var_princ_additional_list = [] # T/F список для хранения информации о том, выполнен ли вариационный принцип или нет (при данном hO и данном Nmax) 
        var_princ_additional_dif_list = [] # список для хранения разницы между энергиями при данном hO при соседних Nmax

        droplist_additional    = []
        dropcounter_additional = 0

        for i in VP_passed_ids:  # цикл по сеткам, которые прошли в. п.
            next = preds_max_Nmax.loc[preds_max_Nmax['annid'] == i]
            previous = preds_max_variational_Nmax.loc[preds_max_variational_Nmax['annid'] == i]
            #---------------------------------
            next = next['Eabs'].to_numpy().reshape(-1, 1)  # оставляем только энергию  
            previous = previous['Eabs'].to_numpy().reshape(-1, 1)          
            next_previous_difference = np.subtract(next, previous) # разница
            #print(f'nex_prev dif = {nex_prev_difference}')
            var_princ_additional_dif_list.extend(next_previous_difference.tolist()) # разница копится для графика
            comparison = next_previous_difference <= VP_EPSILON_2
            var_princ_additional_list.extend(comparison.tolist())
            varprinc_additional = np.prod(comparison)
            #---------------------------------

            if varprinc_additional == 0:  # если вариац. принцип не выполнен, то номер сети добавляется в список для отбрасывания              
                droplist_additional.append(i)                
                dropcounter_additional += 1
    
        droplist_additional = set(droplist_additional)

        var_princ_additional_list = [val[0] for val in var_princ_additional_list]# выше возвращается [[true], [true]...], поэтому [0]
        var_princ_additional_dif_list = [val[0] for val in var_princ_additional_dif_list]

        # гистограмма для разницы энергий
        fig, ax = plt.subplots()
        ax.ticklabel_format(useMathText=True)   
        plt.hist(var_princ_additional_dif_list, edgecolor='black', bins = energy_dif_bins, color = 'mediumaquamarine')
        plt.ylabel('count')
        plt.xlabel('разница по энергии между соседними Nmax')
        plt.title(f'Дополнительная проверка: распределение разницы по энергии. \nКоличество точек: {len(var_princ_additional_dif_list)}, количество сетей: {len(VP_passed_ids)}')    
        plt.savefig(pics_path_e / ('VP_additional_check_dif_hist.' + plot_format), format = plot_format)
        plt.close()

        # гистограмма для разницы энергий (range)
        fig, ax = plt.subplots()
        ax.ticklabel_format(useMathText=True)   
        plt.hist(var_princ_additional_dif_list, edgecolor='black', bins = energy_dif_bins, color = 'mediumaquamarine', range = (a, b))
        plt.ylabel('count')
        plt.xlabel('разница по энергии между соседними Nmax')
        plt.title(f'Дополнительная проверка: распределение разницы по энергии. \nКоличество точек: {len(var_princ_additional_dif_list)}, количество сетей: {len(VP_passed_ids)}')    
        plt.savefig(pics_path_e / ('VP_additional_check_dif_hist_range.' + plot_format), format = plot_format)
        plt.close()

        print(f'VP: additional check: dropcounter = {dropcounter_additional}')
        print(f'VP: additional check: len(droplist) = {len(droplist_additional)}, \n droplist_additional = {droplist_additional}')        
        PP_summary.write(f'\nVP: len(droplist_additional) = {len(droplist_additional)}, \n droplist_additional = {droplist_additional}') 

        VP_passed_ids_additional = set(VP_passed_ids) - droplist_additional # номера сеток, прошедших как основной, так и дополнительный отбор
        VP_passed_ids_additional = sorted(list(VP_passed_ids_additional))      
    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<    


    #print(len(preds_for_VP_plot_ok))
    #print(len(preds_for_VP_plot_true))

    if VP_ADDITIONAL_CHECK == True:
        return VP_passed_ids_additional
    else:
        return VP_passed_ids
    
def straightness_check(preds_energy, preds_radius):
    '''
    проверка на независимость предсказаний от hOmega при большом Nmax

    если разница, между минимумом и максимумом больше определенного числа,
    то такая сетка считается неправильной

    возвращает индексы сетей для энергии и для радиуса, которые прошли проверку
    '''


    # отбрасываем сети на основаниии предсказаний RMS при большом Nmax
    # ============================================
    ids = preds_r['annid'].unique()  # номера нейросетей, которые остались
    #preds_VP_r = preds_r.copy()
    # print(ids)
    # print(preds[['RMS', 'annid']][preds['Nmax'] == n])

    droplist = []  # список сетей для отбрасывания

    # для каждой нейросети вычисляем отклонение diff
    #maxdiff = 0.005  # максимально допустимое отклонение в предсказаниях
    for i in ids:
        rms = preds_r['RMS'][preds_r['annid'] == i][preds_r['Nmax'] == n_r]
        # print(rms)
        diff = abs(rms.max() - rms.min())
        # print(diff)
        if diff > STRAIGHTNESS_TOLERANCE_r:
            droplist.append(i)
    #print(droplist)

    # костыль:
    # если отброшены были почти все сети, то не отбрасываем ничего, чтобы продолжить выполнение программы
    if len(preds_r['annid'].unique()) - len(set(droplist)) < MIN_MODELS_r:
        PP_summary.write('\n радиус: отбор по производной привел к отбрасыванию почти всех сетей, продолжение без отбора.....')
        droplist = []

    # for i in droplist:  # собсно отбрасывание
    #     preds_r.drop(preds_r[preds_r.annid == i].index, inplace=True)
    #preds_VP_r = preds_r[~preds_r['annid'].isin(droplist)]
    #print(len(preds_VP_r['annid'].unique()))  # номера нейросетей, которые остались
    PP_summary.write(f'\n\nрадиус: после отбора по как бы производной осталось {len(preds_r["annid"].unique())} сетей, STRAIGHTNESS_TOLERANCE = {STRAIGHTNESS_TOLERANCE_r}') 
    
    
    ids_passed_r = set(ids) - set(droplist)


    # отбрасываем сети на основаниии предсказаний E при большом Nmax
    # ============================================
    ids = preds_e['annid'].unique()  # номера нейросетей, которые остались
    preds_VP_e = preds_e.copy()
    # print(ids)
    # print(preds[['RMS', 'annid']][preds['Nmax'] == n])

    droplist = []  # список сетей для отбрасывания

    # для каждой нейросети вычисляем отклонение diff    
    for i in ids:
        e = preds_e['Eabs'][preds_e['annid'] == i][preds_e['Nmax'] == n_r]
        # print(rms)
        diff = abs(e.max() - e.min())
        # print(diff)
        if diff > STRAIGHTNESS_TOLERANCE_e:
            droplist.append(i)
    #print(droplist)

    # костыль:
    # если отброшены были почти все сети, то не отбрасываем ничего, чтобы продолжить выполнение программы
    if len(preds_e['annid'].unique()) - len(set(droplist)) < MIN_MODELS_e:
        PP_summary.write('\n энергия: отбор по производной привел к отбрасыванию почти всех сетей, продолжение без отбора.....')
        droplist = []

    # for i in droplist:  # собсно отбрасывание
    #     preds_r.drop(preds_r[preds_r.annid == i].index, inplace=True)
    preds_VP_r = preds_r[~preds_r['annid'].isin(droplist)]
    print(len(preds_VP_r['annid'].unique()))  # номера нейросетей, которые остались
    PP_summary.write(f'\n\n энергия: после отбора по как бы производной осталось {len(preds_e["annid"].unique())} сетей, STRAIGHTNESS_TOLERANCE = {STRAIGHTNESS_TOLERANCE_e}') 
    
    
    ids_passed_e = set(ids) - set(droplist)

    ids_passed_e = sorted(list(ids_passed_e))
    ids_passed_r = sorted(list(ids_passed_r))

    return ids_passed_e, ids_passed_r
    
    # ===========================================

def Deviation(data, reference_data, quantile, bins, plot_title): # Расчет(D)eviation на соотв. данных
    # из data - предсказания
    # из ref data - инофрмация для сравнения
    
    print(f'Deviation: data = {data}')
    print(f'Deviation: reference_data = {reference_data}')   
    
    
    if len(reference_data['Nmax'].unique()) != 1:
        raise ValueError("Deviation: reference_data Nmax is not single")        
    
    col_name = list(reference_data.columns)[-1] # имя колонки (Eabs или RMS)   
    print(f'Deviation: col_name = {col_name}')
    
    
    Nmax_for_metric = reference_data['Nmax'].unique()[0] # определение Nmax из поданного датафрейма
    print(f'Deviation: Nmax_for_metric = {Nmax_for_metric}')
    
    
    
    preds_for_D = data[data['Nmax'] == Nmax_for_metric] # выделение предсказаний для модельного пространства, служащего для расчета метрики
       
    
    preds_for_D_min_hO = preds_for_D.hOmega.min()
    preds_for_D_max_hO = preds_for_D.hOmega.max()
    
    reference_data_min_hO = reference_data.hOmega.min()
    reference_data_max_hO = reference_data.hOmega.max()
    
    
    hO_min = max([preds_for_D_min_hO, reference_data_min_hO])  # минимальное hOmega, которое встречается среди всех данных сразу
    hO_max = min([preds_for_D_max_hO, reference_data_max_hO])  # максимальное hOmega, которое встречается среди всех данных сразу
    #print(hO_min)
    #print(hO_max)

    # унифицируем данные по hOmega
    preds_for_D.drop(preds_for_D[preds_for_D.hOmega < hO_min].index, inplace=True)
    preds_for_D.drop(preds_for_D[preds_for_D.hOmega > hO_max].index, inplace=True)
    

    reference_data.drop(reference_data[reference_data.hOmega < hO_min].index, inplace=True)
    reference_data.drop(reference_data[reference_data.hOmega > hO_max].index, inplace=True)
    
    ref_d_hOmegas = reference_data['hOmega'].unique() # уникальные hOmega в данных для сравнения    
    preds_for_D = preds_for_D[preds_for_D.hOmega.isin(ref_d_hOmegas)] # отбрасывание у предсказаний тех данных, к которым нет аналогичных в исходных данных      
    
    print(preds_for_D[preds_for_D['annid'] == 0])    
    print(reference_data)
    
    sum_of_squared = []    
    D_df = []
    
    # уменьшаемое = предсказания
    # вычитаемое = точные (известные, референсные) значения
    subtrahend = reference_data[col_name].to_numpy() # данные для сравнения (энергия или радиус) в виде массива
    
    # на всякий случай сортировка, хотя порядок должен быть и так правильный
    annids = preds_for_D['annid'].unique()
    annids.sort()    
    #print(f"Derivation: annids = {annids}")
    
    for ann_id in annids:
        minuend = preds_for_D[col_name][preds_for_D['annid'] == ann_id].to_numpy() # выделение одной сети
        #print(forD_arr)
        subtr = np.subtract(minuend, subtrahend) # разность
        #print(subtr)
        summa = (np.sum(np.square(subtr))) # сумма [квадратов разности (поэлементно)]
        sum_of_squared.append(math.sqrt(summa / len(subtr))) # среднее
        D_df.append([math.sqrt(summa / len(subtr)), ann_id])
    
    D_df = pd.DataFrame(data = D_df, columns= ['D', 'annid'])
    #print(D_df)
    
    #print(sum_of_squared)
    #построение гистограммы sum_of_squared
    plt.hist(sum_of_squared, bins = bins, edgecolor='black', color='#00CC99')
    
    median_value = statistics.median(sum_of_squared) # медианное значение
    
    Q = np.quantile(sum_of_squared, quantile) # заданный квантиль
    
    print(f'Deviation: Q1 = {np.quantile(sum_of_squared, 0.25)}')
    
    plot_title = plot_title + f' D для {col_name}; D_median = {format(median_value, ".5f")}'    
    
    plt.axvline(x=median_value, color='black', linestyle='-') # черная линия отмечает медианное значение 
    plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
    plt.xlabel('(D)eviation')
    plt.ylabel('count')
    plt.title(plot_title + f'; сетей: {len(annids)}')
    if col_name == 'Eabs':
        plt.savefig(pics_path_e / ('Распределение Deviation_' + plot_title + '.' + plot_format), format = plot_format)
        plt.close()
    if col_name == 'RMS':
        plt.savefig(pics_path_r / ('Распределение Deviation_' + plot_title + '.' + plot_format), format = plot_format)
        plt.close()
        
    #plt.show()
    
    #return sum_of_squared
    return Q, median_value, D_df

def plotting_combo_func(preds_energy, preds_radius, maxN_energy, maxN_radius, n_energy, n_radius, hist_bins, variational_min, plot_title,        
                        reference_df_e, whole_df_e, reference_df_r, whole_df_r,                        
                        E_exact_value=None, RMS_exact_value=None, E_extrapolation_value=None, plot_limits=None, bins = None):
    '''
    функция для построения графиков и гистограмм, в которой почти всё собрано
    '''
    
    
    if plotting_threading == True:
            tasks = []        
            with concurrent.futures.ProcessPoolExecutor(max_workers = number_of_threads) as executor:
                #предсказания для определенного hOmega
                tasks.append(executor.submit(plot_predictions_for_hOmega, preds_energy, df_e, whole_df_e, pics_path_e, 'Eabs', plot_title))
                tasks.append(executor.submit(plot_predictions_for_hOmega, preds_radius, df_r, whole_df_r, pics_path_r, 'RMS',  plot_title))

                # основные графики, гистограммы   
                tasks.append(executor.submit(E_R_plots_and_histograms,
                    preds_energy, preds_radius, maxN_energy, maxN_radius, n_energy, n_radius, 
                    hist_bins, variational_min, plot_title, 
                    E_exact_value = E_exact_value, E_extrapolation_value = E_extrapolation_value, RMS_exact_value = RMS_exact_value,
                    plot_limits = plot_limits, bins = bins)) 
                
                for Nmax in Nmaxes_data_e: # предсказания для модельных пространств, имеющихся тренировочном датасете
                    print(f'Nmax={Nmax}')
                    tasks.append(executor.submit(plot_predictions_forNmax, preds_energy.copy(), df_e.copy(), reference_df_e.copy(), whole_df_e, Nmax, pics_path_e, 'Eabs', plot_title))
                    
                for Nmax in Nmaxes_data_r:
                    print(f'Nmax={Nmax}')
                    tasks.append(executor.submit(plot_predictions_forNmax, preds_radius.copy(), df_r.copy(), reference_df_r.copy(), whole_df_r, Nmax, pics_path_r, 'RMS',  plot_title))
                    
                #предсказания для выделенного модельного пространства
                tasks.append(executor.submit(plot_predictions_forNmax, preds_energy, df_e, reference_df_e, whole_df_e, Nmax_for_metric_e, pics_path_e, 'Eabs', plot_title))
                tasks.append(executor.submit(plot_predictions_forNmax, preds_radius, df_r, reference_df_r, whole_df_r, Nmax_for_metric_r, pics_path_r, 'RMS',  plot_title))
                                
                
                for Nmax in range(10, n_energy+1, PREDICTIONS_FOR_NMAX_STEP): # предсказания энергии для больших Nmax'ов
                    tasks.append(executor.submit(plot_predictions_forNmax, preds_energy, df_e, reference_df_e, whole_df_e, Nmax, pics_path_e, 'Eabs', plot_title))
                
                # отдельно строится для Nmax = n_energy
                tasks.append(executor.submit(plot_predictions_forNmax, preds_energy, df_e, reference_df_e, whole_df_e, n_energy, pics_path_e, 'Eabs', plot_title))
                    
                
            # без multiprocessing    
            mu_tildeE, sigma_tildeE, mu_tildeRMS, sigma_tildeRMS, mu_E, sigma_E, mu_RMS, sigma_RMS = E_R_plots_and_histograms(
                preds_energy, preds_radius, maxN_energy, maxN_radius, n_energy, n_radius, 
                hist_bins, variational_min, plot_title, 
                E_exact_value = E_exact, E_extrapolation_value=E_extrapolation_b, RMS_exact_value = rms_exact)
                
                
    else:  

        #предсказания для определенного hOmega
        plot_predictions_for_hOmega(preds_energy, df_e, whole_df_e, pics_path_e, 'Eabs', plot_title)
        plot_predictions_for_hOmega(preds_radius, df_r, whole_df_r, pics_path_r, 'RMS',  plot_title)
        
        # предсказания для модельных пространств, имеющихся тренировочном датасете
        for Nmax in Nmaxes_data_e:
            plot_predictions_forNmax(preds_energy, df_e, reference_df_e, whole_df_e, Nmax, pics_path_e, 'Eabs', plot_title)
            
        for Nmax in Nmaxes_data_r:
            plot_predictions_forNmax(preds_radius, df_r, reference_df_r, whole_df_r, Nmax, pics_path_r, 'RMS',  plot_title)
        
        
        
        # предсказания для выделенного модельного пространства
        plot_predictions_forNmax(preds_energy, df_e, reference_df_e, whole_df_e, Nmax_for_metric_e, pics_path_e, 'Eabs', plot_title)
        plot_predictions_forNmax(preds_radius, df_r, reference_df_r, whole_df_r, Nmax_for_metric_e, pics_path_r, 'RMS',  plot_title)  
        
        # предсказания энергии для больших Nmax'ов
        for Nmax in range(10, n_energy, PREDICTIONS_FOR_NMAX_STEP):
            plot_predictions_forNmax(preds_energy, df_e, reference_df_e, whole_df_e, Nmax, pics_path_e, 'Eabs', plot_title)

        # отдельно строится для Nmax = n_energy
        plot_predictions_forNmax(preds_energy, df_e, reference_df_e, whole_df_e, n_energy, pics_path_e, 'Eabs', plot_title)
        
        # основные графики, гистограммы
        E_R_plots_and_histograms(
            preds_energy, preds_radius, maxN_radius, maxN_radius, n_energy, n_radius, 
            20, variational_min, plot_title, 
            E_exact_value = E_exact, E_extrapolation_value=E_extrapolation_b, RMS_exact_value = rms_exact,
            plot_limits=plot_limits, bins = 20)                               
        
        mu_tildeE, sigma_tildeE, mu_tildeRMS, sigma_tildeRMS, mu_E, sigma_E, mu_RMS, sigma_RMS = E_R_plots_and_histograms(
            preds_energy, preds_radius, maxN_radius, maxN_radius, n_energy, n_radius, 
            20, variational_min, plot_title, 
            E_exact_value = E_exact, E_extrapolation_value=E_extrapolation_b, RMS_exact_value = rms_exact)

    # # for bounds_coef in [1, 2, 3, 4, 5]:
    # #     for offset_coef in [0, 1, -1, 2, -2, 3, -3]:
    # #         for binz in [10,15,20]:
    # #             plotting_limits = plot_limits(avgE_plots, stdE_plots, avgRMS_plots, stdRMS_plots, bounds_coef, offset_coef)
    # #             E_R_plots_and_histograms(preds_e, preds_r, maxN_e, maxN_r, n_e, n_r, 
    # #                 20, variational_min, 'после отбора по (D)eviation', 
    # #                 E_exact_value = E_exact, E_extrapolation_value=E_extrapolation_b, RMS_exact_value = rms_exact, 
    # #                 plot_limits = plotting_limits, bins = binz)  

    return mu_tildeE, sigma_tildeE, mu_tildeRMS, sigma_tildeRMS, mu_E, sigma_E, mu_RMS, sigma_RMS

def mse_over_epochs(path, file_name: str, pictures_path):
    '''
    # path      - путь к файлу без его имени
    # file_name - имя файла
    # pictures_path - путь для сохранения картинок

    считывается файл с вычисленным mse после каждой эпохи обучения;

    строится график усредненного по ансамблю mse в в ходе обучения:

    1) вычисляется медианный mse по ансамблю
    2) вычисляются min и max mse после каждой эпохи обучения
    3) вычисляется Q_1 и Q_3 (квартили) mse после каждой эпохи обучения
    '''

    Q1 = 0.25 # первый квартиль
    Q3 = 0.75 # третий квартиль
    window_size = 20 # размер окна для вычисления скользящего среднего

    df = pd.read_csv(path / (file_name + '.csv'), sep='\t', index_col=False)

    processed_df = pd.DataFrame() # обработанный датафрейм    

    epochs = sorted(df['epoch'].unique())
    annids = sorted(df['annid'].unique())

    med_mse_per_epoch = []
    min_mse_per_epoch = []
    max_mse_per_epoch = []
    mse_Q1_per_epoch  = []
    mse_Q3_per_epoch  = []

    # график без усреднения
    # plt.figure()
    # for i in annids:
    #     mse_for_plot = df['mse'].loc[df['annid'] == i]    
    #     plt.plot(epochs, mse_for_plot)
    # plt.savefig('without_averaging.jpg')
    # plt.close()

    # по всему ансамблю значения
    # медианное, мин, макс, Q1 и Q3
    for epoch in epochs:
        if epoch % 100 == 0:
            print(f'mse over epochs: processing - epoch = {epoch}')
        selected_df = df.loc[df['epoch'] == epoch]        
        med_mse_per_epoch.append(selected_df['mse'].median()) # медианный mse
        min_mse_per_epoch.append(selected_df['mse'].min())    
        max_mse_per_epoch.append(selected_df['mse'].max())    
        mse_Q1_per_epoch. append(selected_df['mse'].quantile(q = Q1))
        mse_Q3_per_epoch. append(selected_df['mse'].quantile(q = Q3))


    processed_df['epoch'] = epochs
    processed_df['med_mse_per_epoch'] = med_mse_per_epoch
    processed_df['min_mse_per_epoch'] = min_mse_per_epoch
    processed_df['max_mse_per_epoch'] = max_mse_per_epoch
    processed_df['mse_Q1_per_epoch' ] = mse_Q1_per_epoch
    processed_df['mse_Q3_per_epoch' ] = mse_Q3_per_epoch
    processed_df['mse_iqr_per_epoch'] = np.subtract(np.array(mse_Q3_per_epoch), np.array(mse_Q1_per_epoch)) # вычисление iqr

    #print(processed_df)
    processed_df.to_csv(os.path.join(path / (file_name + '_processed.csv')), sep='\t', index=False, float_format = '%.5e')
    print('mse_over_epochs: processing done')

    # вычисление скользящего среднего для более красивых графиков
    # скользящее среднее может ОЧЕНЬ долго вычисляться
    processed_w_rolling_df = pd.DataFrame()
    processed_w_rolling_df['med_mse_per_epoch'] = processed_df['med_mse_per_epoch'].rolling(window_size, closed = 'both', min_periods = 1).mean()
    processed_w_rolling_df['min_mse_per_epoch'] = processed_df['min_mse_per_epoch'].rolling(window_size, closed = 'both', min_periods = 1).mean()
    processed_w_rolling_df['max_mse_per_epoch'] = processed_df['max_mse_per_epoch'].rolling(window_size, closed = 'both', min_periods = 1).mean()
    processed_w_rolling_df['mse_Q1_per_epoch' ] = processed_df['mse_Q1_per_epoch' ].rolling(window_size, closed = 'both', min_periods = 1).mean()
    processed_w_rolling_df['mse_Q3_per_epoch' ] = processed_df['mse_Q3_per_epoch' ].rolling(window_size, closed = 'both', min_periods = 1).mean()
    processed_w_rolling_df['mse_iqr_per_epoch'] = processed_df['mse_iqr_per_epoch'].rolling(window_size, closed = 'both', min_periods = 1).mean()

    #print(processed_w_rolling_df)    
    processed_w_rolling_df.to_csv(os.path.join(path / (file_name + f'_processed_w_rolling_window{window_size}.csv')), sep='\t', index=False, float_format = '%.5e')

    def mse_over_epochs_plots(df, additional_name: str, pictures_path):
        '''
        df = dataframe

        чисто внутренняя функция для построения графиков

        принимает датафрейм с названиями столбцов:
        med_mse_per_epoch, min_mse_per_epoch, max_mse_per_epoch, mse_Q1_per_epoch, mse_Q3_per_epoch, mse_iqr_per_epoch

        строит графики и сохраняет с названием 'addtional name + прописанное название'

        нужна по сути лишь для того, чтобы построить графики с усреднением и без
        '''    
        #plt.figure(figsize=[10, 6])
        plt.xlabel('epoch')
        plt.ylabel('mse')
        plt.plot(epochs, df.med_mse_per_epoch, linewidth = 4, color = 'green')
        plt.plot(epochs, df.min_mse_per_epoch, linewidth = 4, color = 'blue')
        plt.plot(epochs, df.max_mse_per_epoch, linewidth = 4, color = 'red')
        #plt.fill_between(epochs, avg_minus_std, avg_plus
        # _std, color = 'black', alpha = 0.3)
        plt.plot(epochs, df.mse_Q1_per_epoch, linewidth = 4, linestyle = 'solid', color = 'grey')
        plt.plot(epochs, df.mse_Q3_per_epoch, linewidth = 4, linestyle = 'solid', color = 'grey')
        plt.savefig(pictures_path / ('mse_over_epochs_with_averaging_linear' + additional_name + "." + plot_format), format = plot_format)
        plt.ylim(0, 1e-6)
        plt.savefig(pictures_path / ('mse_over_epochs_with_averaging_linear_limited' + additional_name + "." + plot_format), format = plot_format)
        plt.yscale('log')
        plt.autoscale()
        plt.savefig(pictures_path / ('mse_over_epochs_with_averaging_log' + additional_name + "." +  plot_format), format = plot_format)        
        plt.ylim(1e-9, 1e-4)
        plt.savefig(pictures_path / ('mse_over_epochs_with_averaging_log_limited' + additional_name + "." +  plot_format), format = plot_format)
        plt.close()

        #plt.figure(figsize=[10, 6])
        plt.xlabel('epoch')
        plt.ylabel('mse IQR')
        plt.plot(epochs, df.mse_iqr_per_epoch, linewidth = 4)
        plt.savefig(pictures_path / ('mse_iqr_over_epochs_linear' + additional_name + "." + plot_format), format = plot_format)
        plt.ylim(0, 1e-6)
        plt.savefig(pictures_path / ('mse_iqr_over_epochs_linear_limited' + additional_name + "." + plot_format), format = plot_format)
        plt.autoscale()
        plt.yscale('log')
        plt.savefig(pictures_path / ('mse_iqr_over_epochs_log' + additional_name + "." + plot_format), format = plot_format)
        plt.ylim(1e-9, 1e-4)
        plt.savefig(pictures_path / ('mse_iqr_over_epochs_log_limited' + additional_name + "." + plot_format), format = plot_format)
        #plt.savefig(pics_path_e / ("mse_std_over_epochs" + "." + plot_format), format = plot_format)
        plt.close()


        #plt.figure(figsize=[10, 6])
        plt.xlabel('epoch')
        plt.ylabel('mse')
        plt.plot(epochs, mse_Q1_per_epoch, linewidth = 4)
        plt.plot(epochs, mse_Q3_per_epoch, linewidth = 4)
        plt.savefig(pictures_path / ('mse_Q1_Q3_std_over_epochs_linear' + additional_name + "." + plot_format), format = plot_format)
        plt.ylim(0, 1e-6)
        plt.savefig(pictures_path / ('mse_Q1_Q3_std_over_epochs_linear_limited' + additional_name + "."+ plot_format), format = plot_format)
        plt.autoscale()
        plt.yscale('log')
        plt.savefig(pictures_path / ('mse_Q1_Q3_std_over_epochs_log' + additional_name + "." + plot_format), format = plot_format)
        plt.ylim(1e-9, 1e-4)
        plt.savefig(pictures_path / ('mse_Q1_Q3_std_over_epochs_log_limited' + additional_name + "." + plot_format), format = plot_format)        
        plt.close()

    # собственно построение графиков
    mse_over_epochs_plots(processed_df, '', pictures_path)
    mse_over_epochs_plots(processed_w_rolling_df, '_w_rolling_avg', pictures_path)

def predictions_loss_kde_plot(predictions_e, predictions_r, loss_e, loss_r, additional_name: str, plot_limits = None):
    '''
    график распределения предсказанных значений и лосса: как в статье
    Extrapolation of nuclear structure observables with artificial neural networks
    W. G. Jiang , G. Hagen, and T. Papenbrock (fig. 4)

    то есть плотность при разном loss (loss на последней эпохе обучения) для энергии и радиуса

    для энергии и для радиуса разная предобработка предсказаний

    predictions - датафреймы предсказаний
    loss - датафреймы с колонками loss и annid

    plotting_limits для границ величин энергии и радиуса (т. е. для предсказаний)
    
    additional_name - добавка для имени файла при сохранении
    '''
    print('predictions_loss_kde_plot: start...')

    #------------------------------------------------------------------------------
    # обработка предсказаний для энергии:
    # выбор предсказаний при max(Nmax), определение минимума энергии при max(Nmax)
    # запись в датафрейм
    predictions_maxN_e = predictions_e.loc[predictions_e['Nmax'] == n_e] # выделение предсказаний для максимального Nmax

    predicted_values = [] # окончательные предсказания

    ids = list(sorted(predictions_maxN_e['annid'].unique()))    

    # отбор loss, так, чтобы номера сетей, которые есть в предсказаниях, были и в loss
    loss_e = loss_e[loss_e['annid'].isin(ids)]
    loss_e.sort_values(by = ['annid'], inplace = True)

    for i in ids:
        E = predictions_maxN_e['Eabs'].loc[predictions_maxN_e['annid'] == i] # зависимость E(hO)        
        predicted_values.append(min(E)) # минимум энергии

    predicted_values_e = pd.DataFrame()

    predicted_values_e['loss']  = loss_e['loss'] # x
    predicted_values_e['Eabs']  = predicted_values # y
    predicted_values_e['annid'] = ids # z
    #------------------------------------------------------------------------------



    #------------------------------------------------------------------------------
    # обработка предсказаний для радиуса:
    # выбор предсказаний при max(Nmax), определение среднего значения радиуса при max(Nmax)
    # запись в датафрейм
    predictions_maxN_r = predictions_r.loc[predictions_r['Nmax'] == n_r] # выделение предсказаний для максимального Nmax

    predicted_values = [] # окончательные предсказания

    ids = list(sorted(predictions_maxN_r['annid'].unique()))

    # отбор loss, так, чтобы номера сетей, которые есть в предсказаниях, были и в loss
    loss_r = loss_r[loss_r['annid'].isin(ids)]
    loss_r.sort_values(by = ['annid'], inplace = True)

    for i in ids:
        R = predictions_maxN_r['RMS'].loc[predictions_maxN_r['annid'] == i] # зависимость R(hO)        
        predicted_values.append(R.mean()) # минимум энергии

    predicted_values_r = pd.DataFrame()

    predicted_values_r['loss']  = loss_r['loss'] # x
    predicted_values_r['RMS']   = predicted_values # y
    predicted_values_r['annid'] = ids # z
    #------------------------------------------------------------------------------

    # вычисление границы как 95% квантиля для ограничения построения
    q95_e = loss_e['loss'].quantile(q=0.95, interpolation='linear')
    q95_r = loss_r['loss'].quantile(q=0.95, interpolation='linear')

    # ограничения построения графика
    if plot_limits is not None:
        plot_limits_e = plot_limits[0] 
        plot_limits_r = plot_limits[1]
    else:
        plot_limits_e = None
        plot_limits_r = None

    # графики для энергии с разными цветовыми схемами
    #------------------------------------------------------------------------------------------------------------------------
    # sns.kdeplot(data = predicted_values_e, x = 'loss', y = 'Eabs', fill = True, thresh = 0, levels = 100, cmap = "mako_r")    
    # plt.savefig(pics_path_e / ('kde_plot_mako_r' + additional_name + "." + plot_format), format = plot_format)
    # if plot_limits is not None:
    #     plt.xlim(left = 0.0, right = q95_e)
    #     plt.ylim(plot_limits_e)
    #     plt.savefig(pics_path_e / ('kde_plot_mako_r_limited' + additional_name + "." + plot_format), format = plot_format)
    # plt.close()

    sns.kdeplot(data = predicted_values_e, x = 'loss', y = 'Eabs', cmap="Reds", fill = True)    
    plt.savefig(pics_path_e / ('kde_plot' + additional_name + "." + plot_format), format = plot_format)
    if plot_limits is not None:
        plt.xlim(left = 0.0, right = q95_e)
        plt.ylim(plot_limits_e)
        plt.savefig(pics_path_e / ('kde_plot_limited' + additional_name + "." + plot_format), format = plot_format)
    plt.close()

    sns.kdeplot(data = predicted_values_e, x = 'loss', y = 'Eabs', cmap="Reds", fill = True, bw_adjust = 0.5)    
    plt.savefig(pics_path_e / ('kde_plot0.5' + additional_name + "." + plot_format), format = plot_format)
    if plot_limits is not None:
        plt.xlim(left = 0.0, right = q95_e)
        plt.ylim(plot_limits_e)
        plt.savefig(pics_path_e / ('kde_plot0.5_limited' + additional_name + "." + plot_format), format = plot_format)
    plt.close()

    sns.kdeplot(data = predicted_values_e, x = 'loss', y = 'Eabs', cmap="Reds", fill = True, bw_adjust = 0.7)
    plt.savefig(pics_path_e / ('kde_plot0.7' + additional_name + "." + plot_format), format = plot_format)
    if plot_limits is not None:
        plt.xlim(left = 0.0, right = q95_e)
        plt.ylim(plot_limits_e)
        plt.savefig(pics_path_e / ('kde_plot0.7_limited' + additional_name + "." + plot_format), format = plot_format)
    plt.close()

    sns.kdeplot(data = predicted_values_e, x = 'loss', y = 'Eabs', cmap="Reds", fill = True, bw_adjust = 0.3)
    plt.savefig(pics_path_e / ('kde_plot0.3' + additional_name + "." + plot_format), format = plot_format)
    if plot_limits is not None:
        plt.xlim(left = 0.0, right = q95_e)
        plt.ylim(plot_limits_e)
        plt.savefig(pics_path_e / ('kde_plot0.3_limited' + additional_name + "." + plot_format), format = plot_format)
    plt.close()
    #------------------------------------------------------------------------------------------------------------------------


    # графики для радиуса
    #------------------------------------------------------------------------------------------------------------------------
    # sns.kdeplot(data = predicted_values_r, x = 'loss', y = 'RMS',  fill = True, thresh = 0, levels = 100, cmap = "mako_r")
    # plt.savefig(pics_path_r / ('kde_plot_mako_r' + additional_name + "." + plot_format), format = plot_format)
    # if plot_limits is not None:
    #     plt.xlim(left = 0.0, right = q95_r)
    #     plt.ylim(plot_limits_r)
    #     plt.savefig(pics_path_r / ('kde_plot_mako_r_limited' + additional_name + "." + plot_format), format = plot_format)
    # plt.close()

    sns.kdeplot(data = predicted_values_r, x = 'loss', y = 'RMS', cmap="Reds", fill = True)    
    plt.savefig(pics_path_r / ('kde_plot' + additional_name + "." + plot_format), format = plot_format)
    if plot_limits is not None:
        plt.xlim(left = 0.0, right = q95_r)
        plt.ylim(plot_limits_r)
        plt.savefig(pics_path_r / ('kde_plot_limited' + additional_name + "." + plot_format), format = plot_format)  
    plt.close()

    sns.kdeplot(data = predicted_values_r, x = 'loss', y = 'RMS', cmap="Reds", fill = True, bw_adjust = 0.5)    
    plt.savefig(pics_path_r / ('kde_plot0.5' + additional_name + "." + plot_format), format = plot_format)
    if plot_limits is not None:
        plt.xlim(left = 0.0, right = q95_r)
        plt.ylim(plot_limits_r)
        plt.savefig(pics_path_r / ('kde_plot0.5_limited' + additional_name + "." + plot_format), format = plot_format)  
    plt.close()

    sns.kdeplot(data = predicted_values_r, x = 'loss', y = 'RMS', cmap="Reds", fill = True, bw_adjust = 0.7)
    plt.savefig(pics_path_r / ('kde_plot0.7' + additional_name + "." + plot_format), format = plot_format)
    if plot_limits is not None:
        plt.xlim(left = 0.0, right = q95_r)
        plt.ylim(plot_limits_r)
        plt.savefig(pics_path_r / ('kde_plot0.7_limited' + additional_name + "." + plot_format), format = plot_format) 
    plt.close()

    sns.kdeplot(data = predicted_values_r, x = 'loss', y = 'RMS', cmap="Reds", fill = True, bw_adjust = 0.3)
    plt.savefig(pics_path_r / ('kde_plot0.3' + additional_name + "." + plot_format), format = plot_format)
    if plot_limits is not None:
        plt.xlim(left = 0.0, right = q95_r)
        plt.ylim(plot_limits_r)
        plt.savefig(pics_path_r / ('kde_plot0.3_limited' + additional_name + "." + plot_format), format = plot_format)
    plt.close()
    #------------------------------------------------------------------------------------------------------------------------

    print('predictions_loss_kde_plot: end')


# def plotting_combo_func_for_E(preds_energy, maxN_energy, n_energy, hist_bins, variational_min, plot_title,        
#                             reference_df_e, whole_df_e,  E_exact_value=None,  E_extrapolation_value=None, plot_limits=None, bins = None))


#=====================================
# считывание даты и времени начала из skeleton_summary
with open(pathlib.Path.cwd() / '..' / 'skeleton_summary.txt', 'r') as skeleton_summary:    
    lines = skeleton_summary.readlines()
    for row in lines:        
        target = 'skeleton begin:'
        if row.find(target) == 0:
            #print('string exists in file')
            #print('line Number:', lines.index(row))
            begin_idx = lines.index(row)

begin_str = lines[begin_idx] # строка, содержащая время начала выполнения скрипта
begin_str = begin_str.split('skeleton begin: ')[1] # подстрока с датой
begin_str = begin_str.split('\n')[0]
SKELETON_BEGIN = datetime.datetime.strptime(begin_str,'%Y-%m-%d %H:%M:%S.%f') # время запуска (по сути) skeleton.py
PP_summary.write(f'Skeleton script begin: {SKELETON_BEGIN}\n')


# считывание даты и времени окончания обучения из part_X -> train&predict_summary
# для каждой части отдельно
end_times = []
# поиск строки, содержащей "END"
for part in range(NUM_PARTS):
    with open(pathlib.Path.cwd() / '..' / f'part_{part}' / 'train&predict_summary.txt', encoding = 'utf8') as train_summary:
        lines = train_summary.readlines()
        #print(lines)
        for row in lines:        
            target = 'END'
            if row.find(target) == 0:
                #print('string exists in file')
                #print('line Number:', lines.index(row))
                end_idx = lines.index(row)
    part_end_str = lines[end_idx]                                                     # строка, содержащая время начала выполнения скрипта
    idx_of_datetime_of_string = re.search(r'\d{4}-\d{2}-\d{2}', part_end_str).start() # подстрока с датой
    part_end_str = part_end_str[idx_of_datetime_of_string:]                           # извлечение временного штампа        
    part_end_str = part_end_str.split('\n')[0]                                        # убирает управляющий символ

    part_end_time = datetime.datetime.strptime(part_end_str,'%Y-%m-%d %H:%M:%S.%f')
    PP_summary.write(f'part {part} completion time: {part_end_time}, elapsed time from the skeleton begin: {(part_end_time - SKELETON_BEGIN)} or {(part_end_time - SKELETON_BEGIN).total_seconds()} second(s)\n')
    end_times.append(part_end_time)

# вычисление разницы во времени завершения частей
if NUM_PARTS > 1:
    parts_time_diff = []
    for end_time in end_times:
        parts_time_diff.append((end_time - min(end_times)))

    max_diff_in_time_of_parts = max(parts_time_diff)
    PP_summary.write(f'max difference in time of partial jobs: {max_diff_in_time_of_parts} or {max_diff_in_time_of_parts.total_seconds()} second(s)\n')

TRAINING_END = max(end_times)
PP_summary.write(f'Training end: {TRAINING_END}\n')

# измерение скорости обучения сеток
total_num_of_networks = (numofnns_e + numofnns_r) * NUM_PARTS
train_speed = round(total_num_of_networks * 3600 / float((TRAINING_END - SKELETON_BEGIN).total_seconds()),1)
PP_summary.write(f'Toral number of networks: {total_num_of_networks}\n')
PP_summary.write(f'Train speed: {train_speed} networks per hour in average\n')



# склейка в один файл из задач-частей:
#   предсказаний
#   информации о loss 
#   весов в слоях
if __name__ == '__main__':
    freeze_support()
    with concurrent.futures.ProcessPoolExecutor(max_workers = PP_ntasks) as executor:
        processes = []
        
        # сбор в один файл предскаазний
        processes.append(executor.submit(assemble_loss_or_predictions_from_partial_jobs, 'predictions', 'energy', path_e))
        processes.append(executor.submit(assemble_loss_or_predictions_from_partial_jobs, 'predictions', 'radius', path_r))
        
        # сбор в один файл (mse) в конце обучения
        # раньше вычислялся loss, но т. к. обучение проводится с весами, было принято решение вычисляеть не loss = mse * weight, в mse
        processes.append(executor.submit(assemble_loss_or_predictions_from_partial_jobs, 'loss', 'energy', path_e))
        processes.append(executor.submit(assemble_loss_or_predictions_from_partial_jobs, 'loss', 'radius', path_r))

        # сбор в один файл mse после каждой эпохи
        processes.append(executor.submit(assemble_loss_or_predictions_from_partial_jobs, 'mse_over_epochs_filtered', 'energy', path_e))
        processes.append(executor.submit(assemble_loss_or_predictions_from_partial_jobs, 'mse_over_epochs_filtered', 'radius', path_r))
        
        if splitting == True:
            processes.append(executor.submit(assemble_loss_or_predictions_from_partial_jobs, 'val_loss', 'energy', path_e))
            processes.append(executor.submit(assemble_loss_or_predictions_from_partial_jobs, 'val_loss', 'radius', path_r))  

        
        if WRITE_WEIGHTS == True:
            # сбор в один файл весов между слоями в конце обучения
            processes.append(executor.submit(assemble_loss_or_predictions_from_partial_jobs, 'weights', 'energy', path_e))
            processes.append(executor.submit(assemble_loss_or_predictions_from_partial_jobs, 'weights', 'radius', path_r))

            # сбор в один файл весов между слоями после определенной эпохи обучения
            for epoch_fs in EPOCHS_FOR_SAMPLING:            
                processes.append(executor.submit(assemble_loss_or_predictions_from_partial_jobs, f"epoch{epoch_fs}weights", 'energy', path_e))
                processes.append(executor.submit(assemble_loss_or_predictions_from_partial_jobs, f"epoch{epoch_fs}weights", 'energy', path_r))


#==============================================
#считывание предсказаний из файла, лоссов
preds_e = pd.read_csv(os.path.join(path_e, 'predictions.csv'), sep='\t', index_col=False)
#print(preds_e)
print(len(preds_e['annid'].unique()))

preds_r = pd.read_csv(os.path.join(path_r, 'predictions.csv'), sep='\t', index_col=False)
#print(preds_r)
print(len(preds_r['annid'].unique()))

preds_e.hOmega.round(2)
preds_r.hOmega.round(2)

preds_e = preds_e.astype({"Nmax": int})
preds_r = preds_r.astype({"Nmax": int})


# построение усредненного по ансамблю mse в течение обучения
mse_over_epochs(path_e, 'mse_over_epochs_filtered', pics_path_e)
mse_over_epochs(path_r, 'mse_over_epochs_filtered', pics_path_r)
# ============================================



#=============================================
numofnns_e = len(preds_e['annid'].unique())
numofnns_r = len(preds_r['annid'].unique())

# сравнение числа сеток которое должно быть с тем, что есть в файлике с предсказаниями
numofnns_from_config_e = num_train_params_e.getint('num_of_neural_networks') * NUM_PARTS
numofnns_from_config_r = num_train_params_r.getint('num_of_neural_networks') * NUM_PARTS

if numofnns_e != numofnns_from_config_e or numofnns_r != numofnns_from_config_r:
    raise ValueError('Something wrong with the number of networks, check predictions file and config file')





#чтение исходных данных
df_e = pd.read_excel(path_data_e, sheet_name = 'data', header = 0, usecols = columns_data_e)
df_r = pd.read_excel(path_data_r, sheet_name = 'data', header = 0, usecols = columns_data_r)

# на всякий случай
df_e = df_e.astype({"Nmax": int})
df_r = df_r.astype({"Nmax": int})

print(df_e)
print(df_r)

# датафреймы cо всеми Nmax для сравнения предсказаний при конкретном Nmax
df_allN_e = df_e.copy()
df_allN_r = df_r.copy()

#==============================
df_for_extrapB = df_e.copy()
df_for_extrapB_alldata = df_allN_e.copy()
# ограничение датасета для построения экстраполяцииБ
df_for_extrapB.drop(df_for_extrapB[df_for_extrapB.Nmax > maxN_e].index, inplace=True)


df_for_extrapB.drop(df_for_extrapB[df_for_extrapB.hOmega < hwcutoff_l_e_forExtrapB].index, inplace=True)
df_for_extrapB_alldata.drop(df_for_extrapB_alldata[df_for_extrapB_alldata.hOmega < hwcutoff_l_e_forExtrapB].index, inplace=True)

print(df_for_extrapB)
print(df_for_extrapB_alldata)

E_extrapolation_b, ExtrapB_uncertainty = extrapolation_b(df_for_extrapB)  # минимум энергии, полученный экстраполяцией по трем модельным пространствам    
#E_extrapolation_b_alldata = extrapolation_b(df_for_extrapB_alldata) 
print("Extrapolation B (used dataset): E_inf = ", E_extrapolation_b)


# повторение препроцессинга
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


#обрезание df_allN по hOmega и по минимуму
#df_allN_e.drop(df_allN_e[df_allN_e.hOmega < hwcutoff_l_e].index, inplace=True)
#df_allN_e.drop(df_allN_e[df_allN_e.hOmega > hwcutoff_r_e].index, inplace=True)

    
if cut_on_min == True: # обрезание энергии левее минимума
    cut_left_than_minE(df_e)
    #cut_left_than_minE(df_allN_e)
    
    
if horizontal_cut == True: # обрезание энергии, лежащей выше некоторой E_horizontal
    df_e.drop(df_e[df_e.Eabs > E_horizontal].index, inplace=True) 
    #df_allN_e.drop(df_allN_e[df_allN_e.Eabs > E_horizontal].index, inplace=True)     
    
    


hwcutoff_l_e = df_e['hOmega'].min()  # переопределяем левую границу hOmega

# определяем вариационный минимум здесь, т.е. в ограниченных данных, но без отбрасывания данных для метрики
variational_min = df_e['Eabs'].min()  # переопределяем вариационный минимум
print(variational_min)
PP_summary.write(f'\n{variational_min}\t вариационный минимум в выбранных даннных')



# для радиуса
df_for_metric_r = df_r[df_r['Nmax'] == Nmax_for_metric_r] # для метрики
df_r.drop(df_r[df_r.Nmax == Nmax_for_metric_r].index, inplace=True)


df_r.drop(df_r[df_r.Nmax > maxN_r].index,         inplace=True)  
df_r.drop(df_r[df_r.Nmax < minN_r].index,         inplace=True)
df_r.drop(df_r[df_r.hOmega < hwcutoff_l_r].index, inplace=True)
df_r.drop(df_r[df_r.hOmega > hwcutoff_r_r].index, inplace=True)

    
#обрезание df_allN по hOmega и по минимуму
#df_allN_r.drop(df_allN_r[df_allN_r.hOmega < hwcutoff_l_r].index, inplace=True)
#df_allN_r.drop(df_allN_r[df_allN_r.hOmega > hwcutoff_r_r].index, inplace=True)




neurons_e = len(df_e)
neurons_r = len(df_r)


#print(df_e)
#print(df_r)

    
# df_e.plot(x="hOmega", y="Eabs", kind='scatter', color='tomato', s=240)
# df_r.plot(x="hOmega", y="RMS",  kind='scatter', color='gold',   s=240)
# plt.show()

# используемые данные 
# более темным цветом - для метрики (тестовые данные)
plt.scatter(df_e['hOmega'], df_e['Eabs'], color = 'tomato', s=20)
plt.scatter(df_for_metric_e['hOmega'], df_for_metric_e['Eabs'], color = 'firebrick', s=20)
plt.ylabel('$Eabs$')
plt.xlabel('$\hbar \Omega$')
Nmaxes_used = list(df_e['Nmax'].unique())
Nmaxes_used.extend(list(df_for_metric_e['Nmax'].unique()))
plt.title(f'Nmax = {sorted(Nmaxes_used)}')
plt.savefig(pics_path_e / ('Используемые данные.' + plot_format), format = plot_format)
#plt.show()
plt.close()


plt.scatter(df_r['hOmega'], df_r['RMS'], color = 'gold', s=20)
plt.scatter(df_for_metric_r['hOmega'], df_for_metric_r['RMS'], color = 'goldenrod', s=20)
plt.ylabel('$RMS$')
plt.xlabel('$\hbar \Omega$')
Nmaxes_used = list(df_r['Nmax'].unique())
Nmaxes_used.extend(list(df_for_metric_r['Nmax'].unique()))
plt.title(f'Nmax = {Nmaxes_used}')
plt.savefig(pics_path_r / ('Используемые данные.' + plot_format), format = plot_format)
#plt.show()
plt.close()

del Nmaxes_used
#==============================================
# для обратной трансформации лосса к значениям энергии или радиуса в МэВ или фм в loss_histograms
rev_tr_args_e = (df_e.Eabs.max(), df_e.Eabs.min(), scaler_out_max_e, scaler_out_min_e)
rev_tr_args_r = (df_r.RMS.max(),  df_r.RMS.min(),  scaler_out_max_r, scaler_out_min_r)


if splitting == True:
    train_loss_e = pd.read_csv(path_e / 'loss.csv', sep='\t')
    median_loss_e, annids_loss_e = loss_histograms(train_loss_e, 'loss', path_e, pics_path_e, loss_hist_bins, "energy", reverse_transform_args=rev_tr_args_e)
    train_loss_r = pd.read_csv(path_r / 'loss.csv', sep='\t')
    median_loss_r, annids_loss_r = loss_histograms(train_loss_r, 'loss', path_r, pics_path_r, loss_hist_bins, 'radius', reverse_transform_args=rev_tr_args_r)
    
    valid_loss_e = pd.read_csv(path_e / 'val_loss.csv', sep='\t')
    loss_histograms(valid_loss_e, 'val_loss', path_e, pics_path_e, loss_hist_bins, "energy", reverse_transform_args=rev_tr_args_e)
    valid_loss_r = pd.read_csv(path_r / 'val_loss.csv', sep='\t')
    loss_histograms(valid_loss_r, 'val_loss', path_r, pics_path_r, loss_hist_bins, 'radius', reverse_transform_args=rev_tr_args_r)
    
if splitting == False:    
    train_loss_e = pd.read_csv(path_e / 'loss.csv', sep='\t')
    median_loss_e, annids_loss_e = loss_histograms(train_loss_e, 'loss', path_e, pics_path_e, loss_hist_bins, 'energy', reverse_transform_args=rev_tr_args_e)
    train_loss_r = pd.read_csv(path_r / 'loss.csv', sep='\t')
    median_loss_r, annids_loss_r = loss_histograms(train_loss_r, 'loss', path_r, pics_path_r, loss_hist_bins, 'radius', reverse_transform_args=rev_tr_args_r)
#==================================================================



#==================================================================
# построение гистограмм распределения весов
if WRITE_WEIGHTS == True:
    weights_e = pd.read_csv(path_e / 'weights.csv', sep='\t')
    weights_r = pd.read_csv(path_r / 'weights.csv', sep='\t')
    

    spiciness_coefs_e = weight_histograms(weights_e, path_e, pics_path_e, loss_hist_bins, 'energy')
    spiciness_coefs_r = weight_histograms(weights_r, path_r, pics_path_r, loss_hist_bins, 'radius')
    PP_summary.write(f'\nspiciness_coefs_e = {spiciness_coefs_e}\tЭнергия: коэффициенты остроты для гистограмм весов (без отбора)')
    PP_summary.write(f'\nspiciness_coefs_r = {spiciness_coefs_r}\tРадиус: коэффициенты остроты для гистограмм весов (без отбора)')

    # гистограммы распределения весов для разных эпох
    # словарь датафреймов 
    # плюсом строится гифка
    weights_on_epoch_end_e = {}
    weights_on_epoch_end_r = {}
    #weight_images_e = []
    #weight_images_r = []

    for epoch_fs in EPOCHS_FOR_SAMPLING:
        weights_on_epoch_end_e[epoch_fs] = pd.read_csv(path_e / f'epoch{epoch_fs}weights.csv', sep='\t')
        weights_on_epoch_end_r[epoch_fs] = pd.read_csv(path_r / f'epoch{epoch_fs}weights.csv', sep='\t')

    for epoch_fs in EPOCHS_FOR_SAMPLING:
        weight_histograms(weights_on_epoch_end_e[epoch_fs], path_e, (pics_path_e / 'weight_hists'), loss_hist_bins, f'energy_epoch{epoch_fs}')
        weight_histograms(weights_on_epoch_end_r[epoch_fs], path_r, (pics_path_r / 'weight_hists'), loss_hist_bins, f'radius_epoch{epoch_fs}')
        
            
    ### гифка
    num_of_the_ann_layers = weights_e['layer'].unique() # число слоев считывается из файлика с весами
    num_of_the_ann_layers = sorted(list(num_of_the_ann_layers))

    for layer in num_of_the_ann_layers:
        weight_images_e = []
        weight_images_r = []
        for epoch_fs in EPOCHS_FOR_SAMPLING:
            weight_images_e.append(iio.imread(pics_path_e / 'weight_hists' / f'energy_epoch{epoch_fs}layer_{layer}_weight_distribution.{plot_format}'))
            weight_images_r.append(iio.imread(pics_path_r / 'weight_hists' / f'radius_epoch{epoch_fs}layer_{layer}_weight_distribution.{plot_format}'))
        iio.imwrite(pics_path_e / 'weight_hists' / f'_energy_layer{layer}.gif' , weight_images_e, duration = 500)
        iio.imwrite(pics_path_r / 'weight_hists' / f'_radius_layer{layer}.gif' , weight_images_r, duration = 500)

        #сжатие гифки для экономии размера
        #pgs_optimize(pics_path_e / 'weight_hists' / f'_energy_layer{layer}.gif') # For overwriting the original one
        #pgs_optimize(pics_path_r / 'weight_hists' / f'_radius_layer{layer}.gif') 

    ###

#==================================================================




# запись гиперпараметров и прочего в файл PP_summary
#---------------------------------------------------------------------
PP_summary.write(f'\n{Notes}')

if E_exact is not None:
    PP_summary.write(f'\nE_exact = {E_exact}\t точное значение энергии')
if E_exact is not None:
    PP_summary.write(f'\nrms_exact = {rms_exact}\t точное значение радиуса')
    
PP_summary.write(f'\nvariational_min = {variational_min}\tвариационный минимум энергии в используемых данных')
PP_summary.write(f'\nDeviation_quantile = {Deviation_quantile}\tдля подсчета квантиля при отборе по величине Deviation')

PP_summary.write('\n\nГиперпарамеры')
PP_summary.write('\n----------------------------------------------------')

PP_summary.write(f'\nextra fitting = {EXTRA_FITTING}, дополнительное обучение в течение нескольких эпох с некоторым batch size и learn rate = LR_e(r)')
PP_summary.write(f'\nextra epochs = {EPOCHS_EXTRA}')
PP_summary.write(f'\nbatch size = {BS_EXTRA}')
PP_summary.write(f'\nlearning rate extra coef. = {LR_EXTRA_COEF}, lr_extra  = lr * lr_extra_coef')

PP_summary.write('\nCyclicalLearningRate\n')
PP_summary.write(f'\nCLR_NUM_CYCLES = {CLR_NUM_CYCLES}\t# число циклов в cyclical_learning_rate')
PP_summary.write(f'\nCLR_COEF_MAXLR = {CLR_COEF_MAXLR}\tLR in [base LR, max LR], max LR = CLR_COEF_MAXLR * base LR')



PP_summary.write('\n\nEnergy\n')

PP_summary.write(f'\nE_extrapB = {E_extrapolation_b}\tэнергия, полученная экстраполяцией B по датасету для предсказания энергии')
PP_summary.write(f'\nExtrapB_uncertainty = {ExtrapB_uncertainty}\t погрешность экстраполяции B, определенная как разница между экстраполяцией для старших мод пр-в и значением эстраполяции при младших мод пр-в при оптимальном hO')
#PP_summary.write(f'\nE_extrapB_alldata = {E_extrapolation_b_alldata}\tэнергия, полученная экстраполяцией B по полному датасету')
PP_summary.write(f'\ncut_on_min = {cut_on_min}\tобрезание энергии левее минимума')
PP_summary.write(f'\nhorizontal_cut = {horizontal_cut}\tобрезание энергии, которая лежит выше некоторой E_horizontal')
PP_summary.write(f'\nE_horizontal = {E_horizontal}\tгоризонтальная энергия обрезания')  
PP_summary.write(f'\nAFC_e = {AFC_e}\tactivation function coefficient')
PP_summary.write(f'\nAFC2_e = {AFC2_e}')
PP_summary.write(f'\nAFB_e = {AFB_e}\tactivation function bias')
PP_summary.write(f'\nNE_e = {NE_e}\tnum of epochs')
PP_summary.write(f'\nBS_e = {BS_e}\tbatch size')
PP_summary.write(f'\nLR_e = {LR_e}\tlearning rate')
PP_summary.write(f'\nLRD_e = {LRD_e}\tlearning rate decay')
PP_summary.write(f'\nnumofnns_e = {numofnns_e}\tчисло нейросетей, с которого стартуем')
PP_summary.write(f'\nn_e = {n_e}\tNmax, до которого постепенно будем предсказывать: N = maxN, maxN+2, .. ,n')
PP_summary.write(f'\nn_variational_e = {n_variational_e}\tNmax, до которого проверятется вариационный принцип')

PP_summary.write('\n\nпараметры скейлера (предполагается для minmax scaler):\n')
PP_summary.write(f'\nscaler_in_min_e = {scaler_in_min_e}\tминимум  для входа')
PP_summary.write(f'\nscaler_in_max_e = {scaler_in_max_e}\tмаксимум для входа')
PP_summary.write(f'\nscaler_out_min_e = {scaler_out_min_e}\tминимум  для выхода')
PP_summary.write(f'\nscaler_out_max_e = {scaler_out_max_e}\tмаксимум для выхода\n')
PP_summary.write(f'\nminN_e = {minN_e}\tминимальное учитываемое модельное пространство')
PP_summary.write(f'\nmaxN_e = {maxN_e}\tмаксимальное учитываемое модельное пространство')
PP_summary.write(f'\nhwcutoff_l_e = {hwcutoff_l_e}\tобрезание слева по hOmega')
PP_summary.write(f'\nhwcutoff_r_e = {hwcutoff_r_e}\tобрезание справа по hOmega')


PP_summary.write('\n\nRadius\n')

PP_summary.write(f'\nAFC_r = {AFC_r}\tactivation function coefficient')
PP_summary.write(f'\nAFC2_r = {AFC2_r}')
PP_summary.write(f'\nAFB_r = {AFB_r}\tactivation function bias')
PP_summary.write(f'\nNE_r = {NE_r}\tnum of epochs')
PP_summary.write(f'\nBS_r = {BS_r}\tbatch size')
PP_summary.write(f'\nLR_r = {LR_r}\tlearning rate')
PP_summary.write(f'\nLRD_r = {LRD_r}\tlearning rate decay')
PP_summary.write(f'\nnumofnns_r = {numofnns_r}\tчисло нейросетей, с которого стартуем')
PP_summary.write(f'\nn_r = {n_r}\tNmax, до которого постепенно будем предсказывать: N = maxN, maxN+2, .. ,n')
PP_summary.write('\n\nпараметры скейлера (предполагается для minmax scaler):\n')
PP_summary.write(f'\nscaler_in_min_r = {scaler_in_min_r}\tминимум  для входа')
PP_summary.write(f'\nscaler_in_max_r = {scaler_in_max_r}\tмаксимум для входа')
PP_summary.write(f'\nscaler_out_min_r = {scaler_out_min_r}\tминимум  для выхода')
PP_summary.write(f'\nscaler_out_max_r = {scaler_out_max_r}\tмаксимум для выхода\n')
PP_summary.write(f'\nminN_r = {minN_r}\tминимальное учитываемое модельное пространство')
PP_summary.write(f'\nmaxN_r = {maxN_r}\tмаксимальное учитываемое модельное пространство')
PP_summary.write(f'\nhwcutoff_l_r = {hwcutoff_l_r}\tобрезание слева по hOmega')
PP_summary.write(f'\nhwcutoff_r_r = {hwcutoff_r_r}\tобрезание справа по hOmega')

PP_summary.write('\n----------------------------------------------------')

#запись в summary числа обучаемых параметров
PP_summary.write('\n----------------------------------------------------\nEnergy\n')
#model_e = keras.models.load_model(os.path.join(path_e, str(0)), custom_objects={"scale_fn": scale_fn})
model_e = keras.models.load_model(pathlib.Path.cwd() / '..' / 'part_0' / 'energy' / '0', custom_objects={"scale_fn": scale_fn})
model_e.summary(print_fn=lambda x: PP_summary.write(x + '\n'))
PP_summary.write('\n----------------------------------------------------\nRadius\n')
#model_r = keras.models.load_model(os.path.join(path_r, str(0)), custom_objects={"scale_fn": scale_fn})
model_r = keras.models.load_model(pathlib.Path.cwd() / '..' / 'part_0' / 'radius' / '0', custom_objects={"scale_fn": scale_fn})
model_r.summary(print_fn=lambda x: PP_summary.write(x + '\n'))
PP_summary.write('\n----------------------------------------------------')
#--------------------------------------------------------------------



if __name__ == '__main__':
    freeze_support()        
    Nmaxes_data_e = df_e['Nmax'].unique()
    Nmaxes_data_r = df_r['Nmax'].unique()
    '''
    (ns) = отбор по n сигма среди выбранных
    Построение графиков предсказаний (и прочего) делаюется следующим образом:
        I.  строятся предсказания отобранных по какому-либо признаку, и только по нему (+просто предсказания):
                1) без отбора []
                2) по loss (некая доля сеток с минимальным loss) [loss]
                3) по вариационнному принципу [vp]
                4) по прямоте (сравнивается разница между мин и max значениями с заданным критерием) [straight]
                5) по сходимости предсказаний {дополнительно с более жесткими критериями - x0.1 и с менее жесткими - x10} [conv]
                6) по (D)eviaton [D]
                7) по сходимости предсказаний, НО отбор по сходимости среди прошедших вариационный принцип [convVP]
        
        II. строятся предсказания, отобранные комбинированным образом (как пересечения множеств)
                Энергия:
                8)  2 & 3:                по loss и по вариационному принципу [loss_vp]
                9)  2 & 3 & 4:            по loss, вар. пр. и по прямоте [loss_vp_straight]
                10) 2 & 3 & 4 & (ns):     по loss, вар. пр. и по прямоте и отобранные среди полученных по n sigma [loss_vp_straight_ns]
                11) 3 & 5:                по вариационному принципу и по сходимости [vp_conv]
                12) 3 & 5 & (ns):         по в.п. и сходимости, и после этого отобранные по n sigma [vp_conv_ns]
                13) 3 & 4 & 5:            по в.п., прямости и по сходимости [vp_straight_conv]
                14) 3 & 4 & 5 & (ns):     по в.п., прямости и по сходимости, и после этого отобранные по n sigma [vp_straight_conv_ns]
                15) 2 & 3 & 4 & 6 & (ns): по loss, в. п., прямоте, D и по n sigma среди них [loss_vp_straight_D]
                16) 2 & 7:                по сходимости отобранных по в. п. и по loss [convVP_loss]
                17) 2 & 7 & (ns):         по сходимости отобранных по в. п., по loss и после этого отобранные по n sigma [convVP_loss_ns]
                18) 2 & 7 & 4:            по сходимости отобранных по в. п., по loss и по прямоте [convVP_loss_straight]
                19) 2 & 7 & 4 & (ns):     по сходимости отобранных по в. п., по loss, по прямоте и после этого отобранные по n sigma [loss_vp_straight_conv_ns]
                20) 2 & 3 & 4 & 5:        по loss, в. п., прямоте и сходимости [loss_vp_straight_conv]
                21) 2 & 3 & 4 & 5 & (ns): по loss, в. п., прямоте, сходимости, и после этого отобранные по n sigma [loss_vp_straight_conv_ns]
                Радиус и энергия:
                22) 2 & 4:                по loss и по прямоте [loss_straight]
                23) 2 & 4 & (ns):         по loss по прямоте и после этого отобранные по n sigma [loss_straight_ns]
    '''


    #============================================================================================
    PP_summary.write('\n 1) Без отбора')
    PP_summary.write('\n----------------------------------------------------\n')
    
    
    # список номеров сеток
    annids_e = sorted(preds_e['annid'].unique())
    annids_r = sorted(preds_r['annid'].unique())
    # множества номеров сеток
    set_annids_e = set(annids_e)
    set_annids_r = set(annids_r)

    # предсказания для вcех сеток
    mu_tildeE_raw, sigma_tildeE_raw, mu_tildeRMS_raw, sigma_tildeRMS_raw, mu_E_raw, sigma_E_raw, mu_RMS_raw, sigma_RMS_raw = plotting_combo_func(
        preds_e, preds_r, maxN_e, maxN_r, n_e, n_r,
        20, variational_min, '1_raw_predictions',
        df_for_metric_e, df_allN_e, df_for_metric_r, df_allN_r,
        E_exact, rms_exact, E_extrapolation_b,
        plot_limits = plotting_limits, bins = 20) 

    # расчет (D)eviation
    D_raw_e, D_raw_median_e, D_df_raw_e = Deviation(preds_e, df_for_metric_e, Deviation_quantile, 48, 'предварительный расчет')
    D_raw_r, D_raw_median_r, D_df_raw_r = Deviation(preds_r, df_for_metric_r, Deviation_quantile, 48, 'предварительный расчет')


    PP_summary.write('\n\nЭнергия:\n')
    PP_summary.write(f'\nD_initial_e = {     D_raw_e     }\t предварительный расчет (D)eviation для энергии')
    PP_summary.write(f'\nsigma_tildeE_raw = {sigma_tildeE_raw}\t (S)catter')
    PP_summary.write(f'\nmu_tildeE_raw = {   mu_tildeE_raw   }\t "среднее" значение')
    PP_summary.write(f'\nsigma_E_raw = {     sigma_E_raw     }\t среднеквадратичное отклонение')
    PP_summary.write(f'\nmu_E_raw = {        mu_E_raw        }\t среднее значение')
    PP_summary.write(f"\nannids_raw_e ({len(annids_e)}) = \n{annids_e}")

    PP_summary.write('\n\nРадиус:\n')
    PP_summary.write(f'\n\nD_initial_r = {     D_raw_r       }\t предварительный расчет (D)eviation для радиуса')
    PP_summary.write(f'\nsigma_tildeRMS_raw = {sigma_tildeRMS_raw}\t (S)catter')
    PP_summary.write(f'\nmu_tildeRMS_raw = {   mu_tildeRMS_raw   }\t "среднее" значение')
    PP_summary.write(f'\nsigma_RMS_raw = {     sigma_RMS_raw     }\t среднеквадратичное отклонение')
    PP_summary.write(f'\nmu_RMS_raw = {        mu_RMS_raw        }\t среднее значение')   
    PP_summary.write(f"\n\nannids_raw_r ({len(annids_r)}) = \n{annids_r}") 
    PP_summary.write('\n----------------------------------------------------\n')
    #============================================================================================


    #============================================================================================
    PP_summary.write('\n 2) Отобранные по loss (некая доля сеток с минимальным loss)')
    PP_summary.write('\n----------------------------------------------------\n')

    if len(annids_loss_e) < MIN_MODELS_e:
        print('2) Отбор по loss, энергия: недостаточно сетей, продолжение без отбора.....')
        PP_summary.write('энергия: недостаточно сетей, продолжение без отбора, замена на annids_e.....')
        annids_loss_e = annids_e
    
    if len(annids_loss_r) < MIN_MODELS_r:
        print('2) Отбор по loss, радиус: недостаточно сетей, продолжение без отбора.....')
        PP_summary.write('радиус: недостаточно сетей, продолжение без отбора, замена на annids_r.....')
        annids_loss_r = annids_r
    
    # annids_loss получаеются выше, при построении гистограмм loss'ов
    preds_loss_e = preds_e[preds_e['annid'].isin(annids_loss_e)]
    preds_loss_r = preds_r[preds_r['annid'].isin(annids_loss_r)]

    # множества номеров сеток
    set_annids_loss_e = set(annids_loss_e)
    set_annids_loss_r = set(annids_loss_r)

    

    # предсказания для лучших по loss'у сеток
    mu_tildeE_loss, sigma_tildeE_loss, mu_tildeRMS_loss, sigma_tildeRMS_loss, \
    mu_E_loss,      sigma_E_loss,      mu_RMS_loss,      sigma_RMS_loss = plotting_combo_func(
            preds_loss_e, preds_loss_r, maxN_e, maxN_r, n_e, n_r,
            20, variational_min, f'2_best in loss, q = {LOSS_SELECTION_QUANTILE}',
            df_for_metric_e, df_allN_e, df_for_metric_r, df_allN_r,
            E_exact, rms_exact, E_extrapolation_b,
            plot_limits = plotting_limits, bins = 20)
    
    PP_summary.write('\n\nЭнергия:\n')    
    PP_summary.write(f'\nsigma_tildeE_loss = {sigma_tildeE_loss}\t (S)catter')
    PP_summary.write(f'\nmu_tildeE_loss = {   mu_tildeE_loss   }\t "среднее" значение')
    PP_summary.write(f'\nsigma_E_loss = {     sigma_E_loss     }\t среднеквадратичное отклонение')
    PP_summary.write(f'\nmu_E_loss = {        mu_E_loss        }\t среднее значение')
    PP_summary.write(f"\nannids_loss_e ({len(annids_loss_e)}) = \n{annids_loss_e}")

    PP_summary.write('\n\nРадиус:\n')    
    PP_summary.write(f'\nsigma_tildeRMS_loss = {sigma_tildeRMS_loss}\t (S)catter')
    PP_summary.write(f'\nmu_tildeRMS_loss = {   mu_tildeRMS_loss   }\t "среднее" значение')
    PP_summary.write(f'\nsigma_RMS_loss = {     sigma_RMS_loss     }\t среднеквадратичное отклонение')
    PP_summary.write(f'\nmu_RMS_loss = {        mu_RMS_loss        }\t среднее значение')   
    PP_summary.write(f"\n\nannids_loss_r ({len(annids_loss_r)}) = \n{annids_loss_r}") 
    PP_summary.write('\n----------------------------------------------------\n')
    #==============================================================================================


    #============================================================================================
    PP_summary.write('\n 3) Отобранные по вариационнному принципу')
    PP_summary.write('\n----------------------------------------------------\n')

    annids_vp_e = variational_principle_check(preds_e, df_e) # номера сеток, прошедших вариационный принцип

    if len(annids_vp_e) < MIN_MODELS_e:
        print('3) Отбор по вариационному принципу, энергия: недостаточно сетей, продолжение без отбора.....')
        PP_summary.write('энергия: недостаточно сетей, продолжение без отбора, замена на annids_e.....')
        annids_vp_e = annids_e  
    
    set_annids_vp_e = set(annids_vp_e)

    preds_vp_e = preds_e[preds_e['annid'].isin(annids_vp_e)]

    # предсказания для сеток (для энергии), прошедших вариационный принцип
    # в папке с картинками про радиус создается папка, но в ней картинки просто для preds_r
    mu_tildeE_vp, sigma_tildeE_vp, _, _, \
    mu_E_vp,      sigma_E_vp,      _, _ = plotting_combo_func(
            preds_vp_e, preds_r, maxN_e, maxN_r, n_e, n_r,
            20, variational_min, f'3_vp',
            df_for_metric_e, df_allN_e, df_for_metric_r, df_allN_r,
            E_exact, rms_exact, E_extrapolation_b,
            plot_limits = plotting_limits, bins = 20)

    PP_summary.write('\n\nЭнергия:\n')    
    PP_summary.write(f'\nsigma_tildeE_vp = {sigma_tildeE_vp}\t (S)catter')
    PP_summary.write(f'\nmu_tildeE_vp = {   mu_tildeE_vp   }\t "среднее" значение')
    PP_summary.write(f'\nsigma_E_vp = {     sigma_E_vp     }\t среднеквадратичное отклонение')
    PP_summary.write(f'\nmu_E_vp = {        mu_E_vp        }\t среднее значение')
    PP_summary.write(f"\nannids_vp_e ({len( annids_vp_e)   }) = \n{annids_vp_e}")
    PP_summary.write('\n----------------------------------------------------\n')
    #============================================================================================


    #============================================================================================
    PP_summary.write('\n 4) Отобранные по прямоте (сравнивается разница между мин и max значениями с заданным критерием)')
    PP_summary.write('\n----------------------------------------------------\n')

    annids_straight_e, annids_straight_r = straightness_check(preds_e, preds_r)

    if len(annids_straight_e) < MIN_MODELS_e:
        print('4) Отбор по прямости, энергия: недостаточно сетей, продолжение без отбора.....')
        PP_summary.write('энергия: недостаточно сетей, продолжение без отбора, замена на annids_e.....')
        annids_straight_e = annids_e
    
    if len(annids_straight_r) < MIN_MODELS_r:
        print('4) Отбор по прямости, радиус: недостаточно сетей, продолжение без отбора.....')
        PP_summary.write('радиус: недостаточно сетей, продолжение без отбора, замена на annids_r.....')
        annids_straight_r = annids_r

    preds_straight_e = preds_e[preds_e['annid'].isin(annids_straight_e)]
    preds_straight_r = preds_r[preds_r['annid'].isin(annids_straight_r)]

    # множества номеров сеток
    set_annids_straight_e = set(annids_straight_e)
    set_annids_straight_r = set(annids_straight_r)

    # предсказания сеток, которые дают достаточно "прямые" предсказания
    mu_tildeE_straight, sigma_tildeE_straight, mu_tildeRMS_straight, sigma_tildeRMS_straight, \
    mu_E_straight,      sigma_E_straight,      mu_RMS_straight,      sigma_RMS_straight = plotting_combo_func(
            preds_straight_e, preds_straight_r, maxN_e, maxN_r, n_e, n_r,
            20, variational_min, f'4_straight',
            df_for_metric_e, df_allN_e, df_for_metric_r, df_allN_r,
            E_exact, rms_exact, E_extrapolation_b,
            plot_limits = plotting_limits, bins = 20)
    
    PP_summary.write('\n\nЭнергия:\n')    
    PP_summary.write(f'\nsigma_tildeE_straight = {sigma_tildeE_straight}\t (S)catter')
    PP_summary.write(f'\nmu_tildeE_straight = {   mu_tildeE_straight   }\t "среднее" значение')
    PP_summary.write(f'\nsigma_E_straight = {     sigma_E_straight     }\t среднеквадратичное отклонение')
    PP_summary.write(f'\nmu_E_straight = {        mu_E_straight        }\t среднее значение')
    PP_summary.write(f"\nannids_straight_e ({len(annids_straight_e)}) = \n{annids_straight_e}")

    PP_summary.write('\n\nРадиус:\n')    
    PP_summary.write(f'\nsigma_tildeRMS_straight = {sigma_tildeRMS_straight}\t (S)catter')
    PP_summary.write(f'\nmu_tildeRMS_straight = {   mu_tildeRMS_straight   }\t "среднее" значение')
    PP_summary.write(f'\nsigma_RMS_straight = {     sigma_RMS_straight     }\t среднеквадратичное отклонение')
    PP_summary.write(f'\nmu_RMS_straight = {        mu_RMS_straight        }\t среднее значение')   
    PP_summary.write(f"\n\nannids_straight_r ({len(annids_straight_r)}) = \n{annids_straight_r}") 
    PP_summary.write('\n----------------------------------------------------\n')
    #============================================================================================


    #============================================================================================
    PP_summary.write('\n 5) отобранные по сходимости предсказаний {дополнительно с более жесткими критериями - x0.1 и с менее жесткими - x10}')
    PP_summary.write('\n----------------------------------------------------\n')

    # определение сошедшихся сеток
    median_Nmax_conv_e, annids_conv_e = Nmax_convergence(preds_e, pics_path_e, 40, '', EPS_NMAX, EPS_HO)

    # дополнительные гистограммы о сходимости    
    # более мягкие критерии
    median_Nmax_conv_10_e, annids_conv_10_e = Nmax_convergence(preds_e, pics_path_e, 40, 'дополнительно: ', 10.0 * EPS_NMAX, 10.0 * EPS_HO)
    Nmax_convergence(preds_e, pics_path_e, 40, 'дополнительно: ', 10.0 * EPS_NMAX, 1.0  * EPS_HO)
    Nmax_convergence(preds_e, pics_path_e, 40, 'дополнительно: ', 1.0  * EPS_NMAX, 10.0 * EPS_HO)

    # более жесткие критерии
    Nmax_convergence(preds_e, pics_path_e, 40, 'дополнительно: ', 0.1 *  EPS_NMAX, 1.0  * EPS_HO)
    Nmax_convergence(preds_e, pics_path_e, 40, 'дополнительно: ', 1.0 *  EPS_NMAX, 0.1  * EPS_HO)
    median_Nmax_conv_01_e, annids_conv_01_e = Nmax_convergence(preds_e, pics_path_e, 40, 'дополнительно: ', 0.1 *  EPS_NMAX, 0.1  * EPS_HO)

    if len(annids_conv_e) < MIN_MODELS_e:
        print('5) Отбор по сходимости, энергия: недостаточно сетей, продолжение без отбора.....')
        PP_summary.write('энергия: недостаточно сетей, продолжение без отбора, замена на annids_e.....')
        annids_conv_e = annids_e

    if len(annids_conv_10_e) < MIN_MODELS_e:
        print('5) Отбор по сходимости (x10), энергия: недостаточно сетей, продолжение без отбора.....')
        PP_summary.write('энергия (x10): недостаточно сетей, продолжение без отбора, замена на annids_e.....')
        annids_conv_10_e = annids_e

    if len(annids_conv_01_e) < MIN_MODELS_e:
        print('5) Отбор по сходимости (x10), энергия: недостаточно сетей, продолжение без отбора.....')
        PP_summary.write('энергия (x10): недостаточно сетей, продолжение без отбора, замена на annids_e.....')
        annids_conv_01_e = annids_e

    # множества номеров сеток
    set_annids_conv_e = set(annids_conv_e)
    set_annids_conv_10_e = set(annids_conv_10_e)
    set_annids_conv_01_e = set(annids_conv_01_e)

    preds_conv_e    = preds_e[preds_e['annid'].isin(annids_conv_e)]    
    preds_conv_10_e = preds_e[preds_e['annid'].isin(annids_conv_10_e)]
    preds_conv_01_e = preds_e[preds_e['annid'].isin(annids_conv_01_e)]

    # предсказания сеток после отбора по сходимости
    mu_tildeE_conv, sigma_tildeE_conv, _, _, \
    mu_E_conv,      sigma_E_conv,      _, _ = plotting_combo_func(
            preds_conv_e, preds_r, maxN_e, maxN_r, n_e, n_r,
            20, variational_min, f'5_convergence',
            df_for_metric_e, df_allN_e, df_for_metric_r, df_allN_r,
            E_exact, rms_exact, E_extrapolation_b,
            plot_limits = plotting_limits, bins = 20)
    
    # дополнительные предсказания сеток после отбора по сходимости с менее жесткими критериями
    mu_tildeE_conv_10, sigma_tildeE_conv_10, _, _, \
    mu_E_conv_10,      sigma_E_conv_10,      _, _ = plotting_combo_func(
            preds_conv_10_e, preds_r, maxN_e, maxN_r, n_e, n_r,
            20, variational_min, f'5_convergence_x10',
            df_for_metric_e, df_allN_e, df_for_metric_r, df_allN_r,
            E_exact, rms_exact, E_extrapolation_b,
            plot_limits = plotting_limits, bins = 20)
    
    # дополнительные предсказания сеток после отбора по сходимости с более жесткими критериями
    mu_tildeE_conv_01, sigma_tildeE_conv_01, _, _, \
    mu_E_conv_01,      sigma_E_conv_01,      _, _ = plotting_combo_func(
            preds_conv_01_e, preds_r, maxN_e, maxN_r, n_e, n_r,
            20, variational_min, f'5_convergence_x01',
            df_for_metric_e, df_allN_e, df_for_metric_r, df_allN_r,
            E_exact, rms_exact, E_extrapolation_b,
            plot_limits = plotting_limits, bins = 20)
    
    PP_summary.write('\n\nЭнергия:\n')    
    PP_summary.write(f'\nsigma_tildeE_conv = {sigma_tildeE_conv}\t (S)catter')
    PP_summary.write(f'\nmu_tildeE_conv = {   mu_tildeE_conv   }\t "среднее" значение')
    PP_summary.write(f'\nsigma_E_conv = {     sigma_E_conv     }\t среднеквадратичное отклонение')
    PP_summary.write(f'\nmu_E_conv = {        mu_E_conv        }\t среднее значение')
    PP_summary.write(f"\nannids_conv_e ({len(annids_conv_e)}) = \n{annids_conv_e}")

    PP_summary.write(f'\nsigma_tildeE_conv_10 = {sigma_tildeE_conv_10}\t (S)catter')
    PP_summary.write(f'\nmu_tildeE_conv_10 = {   mu_tildeE_conv_10   }\t "среднее" значение')
    PP_summary.write(f'\nsigma_E_conv_10 = {     sigma_E_conv_10     }\t среднеквадратичное отклонение')
    PP_summary.write(f'\nmu_E_conv_10 = {        mu_E_conv_10        }\t среднее значение')
    PP_summary.write(f"\nannids_conv_10_e ({len(annids_conv_10_e)}) = \n{annids_conv_10_e}")

    PP_summary.write(f'\nsigma_tildeE_conv_01 = {sigma_tildeE_conv_01}\t (S)catter')
    PP_summary.write(f'\nmu_tildeE_conv_01 = {   mu_tildeE_conv_01   }\t "среднее" значение')
    PP_summary.write(f'\nsigma_E_conv_01 = {     sigma_E_conv_01     }\t среднеквадратичное отклонение')
    PP_summary.write(f'\nmu_E_conv_01 = {        mu_E_conv_01        }\t среднее значение')
    PP_summary.write(f"\nannids_conv_01_e ({len(annids_conv_01_e)}) = \n{annids_conv_01_e}")

    # диаграмма перекрытия сеток сошедшихся при разных критериях сходимости            
    plt.figure(figsize=(35, 10))
    supervenn_labels = ['все',            'сошедшиеся',      'сошедшиеся х10',     'сошедшиеся х0.1']
    supervenn(         [set_annids_e, set_annids_conv_e, set_annids_conv_10_e,  set_annids_conv_01_e], supervenn_labels, side_plots=True)
    plt.savefig(pics_path_e / ('conv_intersection_diagram.' + plot_format), format = plot_format)
    plt.close()
    del supervenn_labels 
    PP_summary.write('\n----------------------------------------------------\n')
    #============================================================================================
    

    #============================================================================================
    PP_summary.write('\n 6) отобранные по (D)eviaton')
    PP_summary.write('\n----------------------------------------------------\n')
    # номера сеток, отобраннные на основании величины D из исходных сеток
    ids_to_drop_by_D_e = D_df_raw_e['annid'][D_df_raw_e['D'] > D_raw_e].unique()
    ids_to_drop_by_D_r = D_df_raw_r['annid'][D_df_raw_r['D'] > D_raw_r].unique()   
    

    if len(annids_e) - len(ids_to_drop_by_D_e) < MIN_MODELS_e: # число всех минус выброшенных
        print('6) Отбор по (D)eviaton, энергия: недостаточно сетей, продолжение без отбора.....')
        PP_summary.write('энергия: недостаточно сетей, продолжение без отбора.....')
        ids_to_drop_by_D_e = []

    if len(annids_r) - len(ids_to_drop_by_D_r) < MIN_MODELS_r:
        print('6) Отбор по (D)eviaton, радиус: недостаточно сетей, продолжение без отбора.....')
        PP_summary.write('радиус: недостаточно сетей, продолжение без отбора.....')
        ids_to_drop_by_D_r = []

    preds_D_e = preds_e[~preds_e.annid.isin(ids_to_drop_by_D_e)] # ~ это отрицание
    preds_D_r = preds_r[~preds_r.annid.isin(ids_to_drop_by_D_r)] 

    del ids_to_drop_by_D_e
    del ids_to_drop_by_D_r  

    # списки номеров сеток
    annids_D_e = sorted(preds_D_e['annid'].unique())
    annids_D_r = sorted(preds_D_r['annid'].unique())

    # множества номеров сеток
    set_annids_D_e = set(annids_D_e)
    set_annids_D_r = set(annids_D_r)

    # графики предсказаний для сеток, отобранных по D
    mu_tildeE_D, sigma_tildeE_D, mu_tildeRMS_D, sigma_tildeRMS_D, \
    mu_E_D,      sigma_E_D,      mu_RMS_D,      sigma_RMS_D = plotting_combo_func(
            preds_D_e, preds_D_r, maxN_e, maxN_r, n_e, n_r,
            20, variational_min, f'6_Deviation, q = {Deviation_quantile}',
            df_for_metric_e, df_allN_e, df_for_metric_r, df_allN_r,
            E_exact, rms_exact, E_extrapolation_b,
            plot_limits = plotting_limits, bins = 20)

    PP_summary.write('\n\nЭнергия:\n')    
    PP_summary.write(f'\nsigma_tildeE_D = {sigma_tildeE_D}\t (S)catter')
    PP_summary.write(f'\nmu_tildeE_D = {   mu_tildeE_D   }\t "среднее" значение')
    PP_summary.write(f'\nsigma_E_D = {     sigma_E_D     }\t среднеквадратичное отклонение')
    PP_summary.write(f'\nmu_E_D = {        mu_E_D        }\t среднее значение')
    PP_summary.write(f"\nannids_D_e ({len(annids_D_e)}) = \n{annids_D_e}")

    PP_summary.write('\n\nРадиус:\n')    
    PP_summary.write(f'\nsigma_tildeRMS_D = {sigma_tildeRMS_D}\t (S)catter')
    PP_summary.write(f'\nmu_tildeRMS_D = {   mu_tildeRMS_D   }\t "среднее" значение')
    PP_summary.write(f'\nsigma_RMS_D = {     sigma_RMS_D     }\t среднеквадратичное отклонение')
    PP_summary.write(f'\nmu_RMS_D = {        mu_RMS_D        }\t среднее значение')   
    PP_summary.write(f"\n\nannids_D_r ({len(annids_D_r)}) = \n{annids_D_r}")    
    PP_summary.write('\n----------------------------------------------------\n')
    #============================================================================================


    #============================================================================================
    PP_summary.write('\n 7) отобранные по сходимости среди тех сеток, которые прошли вариационный принцип')
    PP_summary.write('\n----------------------------------------------------\n')    

    # определение сошедшихся сеток
    median_Nmax_convVP_e, annids_convVP_e = Nmax_convergence(preds_vp_e, pics_path_e, 40, 'convVP', EPS_NMAX, EPS_HO)
    
    if len(annids_convVP_e) < MIN_MODELS_e:
        print('7) Отбор по сходимости среди прошедших вариационный принцип, энергия: недостаточно сетей, продолжение без отбора.....')
        PP_summary.write('энергия: недостаточно сетей, продолжение без отбора, замена на annids_e.....')
        annids_convVP_e = annids_e

    # множество номеров сеток
    set_annids_convVP_e = set(annids_convVP_e)    

    preds_convVP_e = preds_e[preds_e['annid'].isin(annids_convVP_e)]      

    # предсказания сеток после отбора по сходимости
    mu_tildeE_convVP, sigma_tildeE_convVP, _, _, \
    mu_E_convVP,      sigma_E_convVP,      _, _ = plotting_combo_func(
            preds_convVP_e, preds_r, maxN_e, maxN_r, n_e, n_r,
            20, variational_min, f'7_converged among passed VP',
            df_for_metric_e, df_allN_e, df_for_metric_r, df_allN_r,
            E_exact, rms_exact, E_extrapolation_b,
            plot_limits = plotting_limits, bins = 20)
    
    PP_summary.write('\n\nЭнергия:\n')    
    PP_summary.write(f'\nsigma_tildeE_convVP = {sigma_tildeE_convVP}\t (S)catter')
    PP_summary.write(f'\nmu_tildeE_convVP = {   mu_tildeE_convVP   }\t "среднее" значение')
    PP_summary.write(f'\nsigma_E_convVP = {     sigma_E_convVP     }\t среднеквадратичное отклонение')
    PP_summary.write(f'\nmu_E_convVP = {        mu_E_convVP        }\t среднее значение')
    PP_summary.write(f"\nannids_convVP_e ({len(annids_convVP_e)}) = \n{annids_convVP_e}")
    PP_summary.write('\n----------------------------------------------------\n')
    #============================================================================================


    #============================================================================================
    PP_summary.write('\n II. строятся предсказания, отобранные комбинированным образом (как пересечения множеств)')
    PP_summary.write('\n Энергия')
    PP_summary.write('\n 8) 2 & 3: отобранные по loss и по вариационному принципу')
    PP_summary.write('\n----------------------------------------------------\n') 

    annids_loss_vp_e = sorted(list(set_annids_loss_e.intersection(set_annids_vp_e)))

    if len(annids_loss_vp_e) < MIN_MODELS_e:
        print('8) 2 & 3: отобранные по loss и по вариационному принципу, энергия: недостаточно сетей, продолжение без отбора.....')
        PP_summary.write('энергия: недостаточно сетей, продолжение без отбора, замена на annids_e.....')
        annids_loss_vp_e = annids_e

    set_annids_loss_vp_e = set(annids_loss_vp_e)

    preds_loss_vp_e = preds_e[preds_e['annid'].isin(annids_loss_vp_e)]

    # графики предсказаний
    mu_tildeE_loss_vp, sigma_tildeE_loss_vp, _, _, \
    mu_E_loss_vp,      sigma_E_loss_vp,      _, _ = plotting_combo_func(
            preds_loss_vp_e, preds_r, maxN_e, maxN_r, n_e, n_r,
            20, variational_min, f'8_loss_vp',
            df_for_metric_e, df_allN_e, df_for_metric_r, df_allN_r,
            E_exact, rms_exact, E_extrapolation_b,
            plot_limits = plotting_limits, bins = 20)

    PP_summary.write('\n\nЭнергия:\n')    
    PP_summary.write(f'\nsigma_tildeE_loss_vp = {sigma_tildeE_loss_vp}\t (S)catter')
    PP_summary.write(f'\nmu_tildeE_loss_vp = {   mu_tildeE_loss_vp   }\t "среднее" значение')
    PP_summary.write(f'\nsigma_E_loss_vp = {     sigma_E_loss_vp     }\t среднеквадратичное отклонение')
    PP_summary.write(f'\nmu_E_loss_vp = {        mu_E_loss_vp        }\t среднее значение')
    PP_summary.write(f"\nannids_loss_vp_e ({len(annids_loss_vp_e)}) = \n{annids_loss_vp_e}")
    PP_summary.write('\n----------------------------------------------------\n')
    #============================================================================================


    #============================================================================================
    PP_summary.write('\n 9) 2 & 3 & 4: Отобранные по loss, вар. пр. и по прямоте')
    PP_summary.write('\n----------------------------------------------------\n') 

    annids_loss_vp_straight_e = sorted(list(set_annids_loss_e.intersection(set_annids_vp_e, set_annids_straight_e)))

    if len(annids_loss_vp_straight_e) < MIN_MODELS_e:
        print('9) 2 & 3: отобранные по loss и по вариационному принципу, энергия: недостаточно сетей, продолжение без отбора.....')
        PP_summary.write('энергия: недостаточно сетей, продолжение без отбора, замена на annids_e.....')
        annids_loss_vp_straight_e = annids_e

    set_annids_loss_vp_straight_e = set(annids_loss_vp_straight_e)

    preds_loss_vp_straight_e = preds_e[preds_e['annid'].isin(annids_loss_vp_straight_e)]

    # графики предсказаний
    mu_tildeE_loss_vp_straight, sigma_tildeE_loss_vp_straight, _, _, \
    mu_E_loss_vp_straight,      sigma_E_loss_vp_straight,      _, _ = plotting_combo_func(
            preds_loss_vp_straight_e, preds_r, maxN_e, maxN_r, n_e, n_r,
            20, variational_min, f'9_loss_vp_straight',
            df_for_metric_e, df_allN_e, df_for_metric_r, df_allN_r,
            E_exact, rms_exact, E_extrapolation_b,
            plot_limits = plotting_limits, bins = 20)

    PP_summary.write('\n\nЭнергия:\n')    
    PP_summary.write(f'\nsigma_tildeE_loss_vp_straight = {sigma_tildeE_loss_vp_straight}\t (S)catter')
    PP_summary.write(f'\nmu_tildeE_loss_vp_straight = {   mu_tildeE_loss_vp_straight   }\t "среднее" значение')
    PP_summary.write(f'\nsigma_E_loss_vp_straight = {     sigma_E_loss_vp_straight     }\t среднеквадратичное отклонение')
    PP_summary.write(f'\nmu_E_loss_vp_straight = {        mu_E_loss_vp_straight        }\t среднее значение')
    PP_summary.write(f"\nannids_loss_vp_straight_e ({len(annids_loss_vp_straight_e)}) = \n{annids_loss_vp_straight_e}")
    PP_summary.write('\n----------------------------------------------------\n')
    #============================================================================================


    #============================================================================================
    PP_summary.write('\n 10) 2 & 3 & 4 & (ns): по loss, вар. пр. и по прямоте и отобранные среди полученных по n sigma')
    PP_summary.write('\n----------------------------------------------------\n') 

    preds_loss_vp_straight_ns_e, _  = iterative_outlier_filtering_of_predictions(preds_loss_vp_straight_e, preds_r, method = iterative_outlier_filtering_method)
    annids_loss_vp_straight_ns_e = sorted(preds_loss_vp_straight_ns_e['annid'].unique())

    if len(annids_loss_vp_straight_ns_e) < MIN_MODELS_e:
        print('10) 2 & 3 & 4 & (ns): по loss, вар. пр. и по прямоте и отобранные среди полученных по n sigma, энергия: недостаточно сетей, продолжение без отбора.....')
        PP_summary.write('энергия: недостаточно сетей, продолжение без отбора, замена на annids_e.....')
        annids_loss_vp_straight_ns_e = annids_e
        preds_loss_vp_straight_ns_e  = preds_e

    set_annids_loss_vp_straight_ns_e = set(annids_loss_vp_straight_ns_e)

    # графики предсказаний
    mu_tildeE_loss_vp_straight_ns, sigma_tildeE_loss_vp_straight_ns, _, _, \
    mu_E_loss_vp_straight_ns,      sigma_E_loss_vp_straight_ns,      _, _ = plotting_combo_func(
            preds_loss_vp_straight_ns_e, preds_r, maxN_e, maxN_r, n_e, n_r,
            20, variational_min, f'10_loss_vp_straight_nsigma',
            df_for_metric_e, df_allN_e, df_for_metric_r, df_allN_r,
            E_exact, rms_exact, E_extrapolation_b,
            plot_limits = plotting_limits, bins = 20)

    PP_summary.write('\n\nЭнергия:\n')    
    PP_summary.write(f'\nsigma_tildeE_loss_vp_straight_ns = {sigma_tildeE_loss_vp_straight_ns}\t (S)catter')
    PP_summary.write(f'\nmu_tildeE_loss_vp_straight_ns = {   mu_tildeE_loss_vp_straight_ns   }\t "среднее" значение')
    PP_summary.write(f'\nsigma_E_loss_vp_straight_ns = {     sigma_E_loss_vp_straight_ns     }\t среднеквадратичное отклонение')
    PP_summary.write(f'\nmu_E_loss_vp_straight_ns = {        mu_E_loss_vp_straight_ns        }\t среднее значение')
    PP_summary.write(f"\nannids_loss_vp_straight_ns_e ({len(annids_loss_vp_straight_ns_e)}) = \n{annids_loss_vp_straight_ns_e}")
    PP_summary.write('\n----------------------------------------------------\n')
    #============================================================================================
    

    #============================================================================================
    PP_summary.write('\n 11) 3 & 5: вариационному принципу и по сходимости')
    PP_summary.write('\n----------------------------------------------------\n') 

    annids_vp_conv_e = sorted(list(set_annids_vp_e.intersection(set_annids_conv_e)))

    if len(annids_vp_conv_e) < MIN_MODELS_e:
        print('11) 3 & 5: отобранные по 3 & 5: вариационному принципу и по сходимости, энергия: недостаточно сетей, продолжение без отбора.....')
        PP_summary.write('энергия: недостаточно сетей, продолжение без отбора, замена на annids_e.....')
        annids_vp_conv_e = annids_e

    set_annids_vp_conv_e = set(annids_vp_conv_e)

    preds_vp_conv_e = preds_e[preds_e['annid'].isin(annids_vp_conv_e)]

    # графики предсказаний
    mu_tildeE_vp_conv, sigma_tildeE_vp_conv, _, _, \
    mu_E_vp_conv,      sigma_E_vp_conv,      _, _ = plotting_combo_func(
            preds_vp_conv_e, preds_r, maxN_e, maxN_r, n_e, n_r,
            20, variational_min, f'11_vp_conv',
            df_for_metric_e, df_allN_e, df_for_metric_r, df_allN_r,
            E_exact, rms_exact, E_extrapolation_b,
            plot_limits = plotting_limits, bins = 20)

    PP_summary.write('\n\nЭнергия:\n')    
    PP_summary.write(f'\nsigma_tildeE_vp_conv = {sigma_tildeE_vp_conv}\t (S)catter')
    PP_summary.write(f'\nmu_tildeE_vp_conv = {   mu_tildeE_vp_conv   }\t "среднее" значение')
    PP_summary.write(f'\nsigma_E_vp_conv = {     sigma_E_vp_conv     }\t среднеквадратичное отклонение')
    PP_summary.write(f'\nmu_E_vp_conv = {        mu_E_vp_conv        }\t среднее значение')
    PP_summary.write(f"\nannids_vp_conv_e ({len(annids_vp_conv_e)}) = \n{annids_vp_conv_e}")
    PP_summary.write('\n----------------------------------------------------\n')
    #============================================================================================


    #============================================================================================
    PP_summary.write('\n 12) 3 & 5 & (ns): отобранные по по в.п. и сходимости, и после этого отобранные по n sigma')
    PP_summary.write('\n----------------------------------------------------\n') 

    preds_vp_conv_ns_e, _  = iterative_outlier_filtering_of_predictions(preds_vp_conv_e, preds_r, method = iterative_outlier_filtering_method)
    annids_vp_conv_ns_e = sorted(preds_vp_conv_ns_e['annid'].unique())

    if len(annids_vp_conv_ns_e) < MIN_MODELS_e:
        print('12) 3 & 5 & (ns): отобранные по по в.п. и сходимости, и после этого отобранные по n sigma, энергия: недостаточно сетей, продолжение без отбора.....')
        PP_summary.write('энергия: недостаточно сетей, продолжение без отбора, замена на annids_e.....')
        annids_vp_conv_ns_e = annids_e
        preds_vp_conv_ns_e  = preds_e

    set_annids_vp_conv_ns_e = set(annids_vp_conv_ns_e)

    # графики предсказаний
    mu_tildeE_vp_conv_ns, sigma_tildeE_vp_conv_ns, _, _, \
    mu_E_vp_conv_ns,      sigma_E_vp_conv_ns,      _, _ = plotting_combo_func(
            preds_vp_conv_ns_e, preds_r, maxN_e, maxN_r, n_e, n_r,
            20, variational_min, f'12_vp_conv_nsigma',
            df_for_metric_e, df_allN_e, df_for_metric_r, df_allN_r,
            E_exact, rms_exact, E_extrapolation_b,
            plot_limits = plotting_limits, bins = 20)

    PP_summary.write('\n\nЭнергия:\n')    
    PP_summary.write(f'\nsigma_tildeE_vp_conv_ns = {sigma_tildeE_vp_conv_ns}\t (S)catter')
    PP_summary.write(f'\nmu_tildeE_vp_conv_ns = {   mu_tildeE_vp_conv_ns   }\t "среднее" значение')
    PP_summary.write(f'\nsigma_E_vp_conv_ns = {     sigma_E_vp_conv_ns     }\t среднеквадратичное отклонение')
    PP_summary.write(f'\nmu_E_vp_conv_ns = {        mu_E_vp_conv_ns        }\t среднее значение')
    PP_summary.write(f"\nannids_vp_conv_ns_e ({len(annids_vp_conv_ns_e)}) = \n{annids_vp_conv_ns_e}")
    PP_summary.write('\n----------------------------------------------------\n')
    #============================================================================================

    
    #============================================================================================
    PP_summary.write('\n 13) 3 & 4 & 5: отобранные по в.п., прямости и по сходимости')
    PP_summary.write('\n----------------------------------------------------\n') 

    annids_vp_straight_conv_e = sorted(list(set_annids_vp_e.intersection(set_annids_straight_e, set_annids_conv_e)))

    if len(annids_vp_straight_conv_e) < MIN_MODELS_e:
        print('13) 3 & 4 & 5: отобранные по в.п., прямости и по сходимости, энергия: недостаточно сетей, продолжение без отбора.....')
        PP_summary.write('энергия: недостаточно сетей, продолжение без отбора, замена на annids_e.....')
        annids_vp_straight_conv_e = annids_e

    set_annids_vp_straight_conv_e = set(annids_vp_straight_conv_e)

    preds_vp_straight_conv_e = preds_e[preds_e['annid'].isin(annids_vp_straight_conv_e)]

    # графики предсказаний
    mu_tildeE_vp_straight_conv, sigma_tildeE_vp_straight_conv, _, _, \
    mu_E_vp_straight_conv,      sigma_E_vp_straight_conv,      _, _ = plotting_combo_func(
            preds_vp_straight_conv_e, preds_r, maxN_e, maxN_r, n_e, n_r,
            20, variational_min, f'13_vp_straight_conv',
            df_for_metric_e, df_allN_e, df_for_metric_r, df_allN_r,
            E_exact, rms_exact, E_extrapolation_b,
            plot_limits = plotting_limits, bins = 20)

    PP_summary.write('\n\nЭнергия:\n')    
    PP_summary.write(f'\nsigma_tildeE_vp_straight_conv = {sigma_tildeE_vp_straight_conv}\t (S)catter')
    PP_summary.write(f'\nmu_tildeE_vp_straight_conv = {   mu_tildeE_vp_straight_conv   }\t "среднее" значение')
    PP_summary.write(f'\nsigma_E_vp_straight_conv = {     sigma_E_vp_straight_conv     }\t среднеквадратичное отклонение')
    PP_summary.write(f'\nmu_E_vp_straight_conv = {        mu_E_vp_straight_conv        }\t среднее значение')
    PP_summary.write(f"\nannids_vp_straight_conv_e ({len(annids_vp_straight_conv_e)}) = \n{annids_vp_straight_conv_e}")
    PP_summary.write('\n----------------------------------------------------\n')
    #============================================================================================


    #============================================================================================
    PP_summary.write('\n 14) 3 & 4 & 5 & (ns): отобранные по в.п., прямости и по сходимости, и после этого отобранные по n sigma')
    PP_summary.write('\n----------------------------------------------------\n') 

    preds_vp_straight_conv_ns_e, _  = iterative_outlier_filtering_of_predictions(preds_vp_straight_conv_e, preds_r, method = iterative_outlier_filtering_method)
    annids_vp_straight_conv_ns_e = sorted(preds_vp_straight_conv_ns_e['annid'].unique())

    if len(annids_vp_straight_conv_ns_e) < MIN_MODELS_e:
        print('14) 3 & 4 & 5 & (ns): отобранные по по в.п. и сходимости, и после этого отобранные по n sigma, энергия: недостаточно сетей, продолжение без отбора.....')
        PP_summary.write('энергия: недостаточно сетей, продолжение без отбора, замена на annids_e.....')
        annids_vp_straight_conv_ns_e = annids_e
        preds_vp_straight_conv_ns_e  = preds_e

    set_annids_vp_straight_conv_ns_e = set(annids_vp_straight_conv_ns_e)

    # графики предсказаний
    mu_tildeE_vp_straight_conv_ns, sigma_tildeE_vp_straight_conv_ns, _, _, \
    mu_E_vp_straight_conv_ns,      sigma_E_vp_straight_conv_ns,      _, _ = plotting_combo_func(
            preds_vp_straight_conv_ns_e, preds_r, maxN_e, maxN_r, n_e, n_r,
            20, variational_min, f'14_vp_straight_conv_nsigma',
            df_for_metric_e, df_allN_e, df_for_metric_r, df_allN_r,
            E_exact, rms_exact, E_extrapolation_b,
            plot_limits = plotting_limits, bins = 20)

    PP_summary.write('\n\nЭнергия:\n')    
    PP_summary.write(f'\nsigma_tildeE_vp_straight_conv_ns = {sigma_tildeE_vp_straight_conv_ns}\t (S)catter')
    PP_summary.write(f'\nmu_tildeE_vp_straight_conv_ns = {   mu_tildeE_vp_straight_conv_ns   }\t "среднее" значение')
    PP_summary.write(f'\nsigma_E_vp_straight_conv_ns = {     sigma_E_vp_straight_conv_ns     }\t среднеквадратичное отклонение')
    PP_summary.write(f'\nmu_E_vp_straight_conv_ns = {        mu_E_vp_straight_conv_ns        }\t среднее значение')
    PP_summary.write(f"\nannids_vp_straight_conv_ns_e ({len(annids_vp_straight_conv_ns_e)}) = \n{annids_vp_straight_conv_ns_e}")
    PP_summary.write('\n----------------------------------------------------\n')
    #============================================================================================


    #============================================================================================
    PP_summary.write('\n 15) 2 & 3 & 4 & 6 & (ns): по loss, в. п., прямоте, D и по n sigma среди них')
    PP_summary.write('\n----------------------------------------------------\n') 

    # без отбора по n sigma
    annids_loss_vp_straight_D_e = sorted(list(set_annids_loss_e.intersection(set_annids_vp_e, set_annids_straight_e, set_annids_D_e)))   

    preds_loss_vp_straight_D_e = preds_e[preds_e['annid'].isin(annids_loss_vp_straight_D_e)]

    #отбор по n sigma
    preds_loss_vp_straight_D_ns_e, _ = iterative_outlier_filtering_of_predictions(preds_loss_vp_straight_D_e, preds_r, method = iterative_outlier_filtering_method)

    annids_loss_vp_straight_D_ns_e = sorted(preds_loss_vp_straight_D_ns_e['annid'].unique())

    if len(annids_loss_vp_straight_D_ns_e) < MIN_MODELS_e:
        print('15) 3 & 4 & 5: отобранные по в.п., прямости и по сходимости, энергия: недостаточно сетей, продолжение без отбора.....')
        PP_summary.write('энергия: недостаточно сетей, продолжение без отбора, замена на annids__e.....')
        preds_loss_vp_straight_D_ns_e =  preds_e
        annids_loss_vp_straight_D_ns_e = annids_e

    set_annids_loss_vp_straight_D_ns_e = set(annids_loss_vp_straight_D_ns_e)

    # графики предсказаний
    mu_tildeE_loss_vp_straight_D_ns, sigma_tildeE_loss_vp_straight_D_ns, _, _, \
    mu_E_loss_vp_straight_D_ns,      sigma_E_loss_vp_straight_D_ns,      _, _ = plotting_combo_func(
            preds_loss_vp_straight_D_ns_e, preds_r, maxN_e, maxN_r, n_e, n_r,
            20, variational_min, f'15_loss_vp_straight_D_ns',
            df_for_metric_e, df_allN_e, df_for_metric_r, df_allN_r,
            E_exact, rms_exact, E_extrapolation_b,
            plot_limits = plotting_limits, bins = 20)

    PP_summary.write('\n\nЭнергия:\n')    
    PP_summary.write(f'\nsigma_tildeE_loss_vp_straight_D_ns = {sigma_tildeE_loss_vp_straight_D_ns}\t (S)catter')
    PP_summary.write(f'\nmu_tildeE_loss_vp_straight_D_ns = {   mu_tildeE_loss_vp_straight_D_ns   }\t "среднее" значение')
    PP_summary.write(f'\nsigma_E_loss_vp_straight_D_ns = {     sigma_E_loss_vp_straight_D_ns     }\t среднеквадратичное отклонение')
    PP_summary.write(f'\nmu_E_loss_vp_straight_D_ns = {        mu_E_loss_vp_straight_D_ns        }\t среднее значение')
    PP_summary.write(f"\nannids_loss_vp_straight_D_ns_e ({len(annids_loss_vp_straight_D_ns_e)}) = \n{annids_loss_vp_straight_D_ns_e}")
    PP_summary.write('\n----------------------------------------------------\n')
    #============================================================================================


    #============================================================================================
    PP_summary.write('\n 16) 2 & 7: отобранные по сходимости предварительно отобранных по в. п. и по loss')
    PP_summary.write('\n----------------------------------------------------\n') 

    annids_convVP_loss_e = sorted(list(set_annids_convVP_e.intersection(set_annids_loss_e)))

    if len(annids_convVP_loss_e) < MIN_MODELS_e:
        print('16) 2 & 7: отобранные по сходимости предварительно отобранных по в. п. и по loss, энергия: недостаточно сетей, продолжение без отбора.....')
        PP_summary.write('энергия: недостаточно сетей, продолжение без отбора, замена на annids_e.....')
        annids_convVP_loss_e = annids_e

    set_annids_convVP_loss_e = set(annids_convVP_loss_e)

    preds_convVP_loss_e = preds_e[preds_e['annid'].isin(annids_convVP_loss_e)]

    # графики предсказаний
    mu_tildeE_convVP_loss, sigma_tildeE_convVP_loss, _, _, \
    mu_E_convVP_loss,      sigma_E_convVP_loss,      _, _ = plotting_combo_func(
            preds_convVP_loss_e, preds_r, maxN_e, maxN_r, n_e, n_r,
            20, variational_min, f'16_convVP_loss',
            df_for_metric_e, df_allN_e, df_for_metric_r, df_allN_r,
            E_exact, rms_exact, E_extrapolation_b,
            plot_limits = plotting_limits, bins = 20)

    PP_summary.write('\n\nЭнергия:\n')    
    PP_summary.write(f'\nsigma_tildeE_convVP_loss = {sigma_tildeE_convVP_loss}\t (S)catter')
    PP_summary.write(f'\nmu_tildeE_convVP_loss = {   mu_tildeE_convVP_loss   }\t "среднее" значение')
    PP_summary.write(f'\nsigma_E_convVP_loss = {     sigma_E_convVP_loss     }\t среднеквадратичное отклонение')
    PP_summary.write(f'\nmu_E_convVP_loss = {        mu_E_convVP_loss        }\t среднее значение')
    PP_summary.write(f"\nannids_convVP_loss_e ({len(annids_convVP_loss_e)}) = \n{annids_convVP_loss_e}")
    PP_summary.write('\n----------------------------------------------------\n')
    #============================================================================================


    #============================================================================================
    PP_summary.write('\n 17) 2 & 7 & (ns): по сходимости отобранных по в. п., по loss и после этого отобранные по n sigma')
    PP_summary.write('\n----------------------------------------------------\n') 

    #отбор по n sigma
    preds_convVP_loss_ns_e, _ = iterative_outlier_filtering_of_predictions(preds_convVP_loss_e, preds_r, method = iterative_outlier_filtering_method)

    annids_convVP_loss_ns_e = sorted(preds_convVP_loss_ns_e['annid'].unique())

    if len(annids_convVP_loss_ns_e) < MIN_MODELS_e:
        print('17) 2 & 7 & (ns): по сходимости отобранных по в. п., по loss и после этого отобранные по n sigma, энергия: недостаточно сетей, продолжение без отбора.....')
        PP_summary.write('энергия: недостаточно сетей, продолжение без отбора, замена на annids_e.....')
        preds_convVP_loss_ns_e  = preds_e
        annids_convVP_loss_ns_e = annids_e

    set_annids_convVP_loss_ns_e = set(annids_convVP_loss_ns_e)

    # графики предсказаний
    mu_tildeE_convVP_loss_ns, sigma_tildeE_convVP_loss_ns, _, _, \
    mu_E_convVP_loss_ns,      sigma_E_convVP_loss_ns,      _, _ = plotting_combo_func(
            preds_convVP_loss_ns_e, preds_r, maxN_e, maxN_r, n_e, n_r,
            20, variational_min, f'17_convVP_loss_ns',
            df_for_metric_e, df_allN_e, df_for_metric_r, df_allN_r,
            E_exact, rms_exact, E_extrapolation_b,
            plot_limits = plotting_limits, bins = 20)

    PP_summary.write('\n\nЭнергия:\n')    
    PP_summary.write(f'\nsigma_tildeE_convVP_loss_ns = {sigma_tildeE_convVP_loss_ns}\t (S)catter')
    PP_summary.write(f'\nmu_tildeE_convVP_loss_ns = {   mu_tildeE_convVP_loss_ns   }\t "среднее" значение')
    PP_summary.write(f'\nsigma_E_convVP_loss_ns = {     sigma_E_convVP_loss_ns     }\t среднеквадратичное отклонение')
    PP_summary.write(f'\nmu_E_convVP_loss_ns = {        mu_E_convVP_loss_ns        }\t среднее значение')
    PP_summary.write(f"\nannids_convVP_loss_ns_e ({len(annids_convVP_loss_ns_e)}) = \n{annids_convVP_loss_ns_e}")
    PP_summary.write('\n----------------------------------------------------\n')    
    #============================================================================================


    #============================================================================================
    PP_summary.write('\n 18) 2 & 7 & 4: отобранные по сходимости предварительно отобранных по в. п., по loss и по прямоте')
    PP_summary.write('\n----------------------------------------------------\n') 
    annids_convVP_loss_straight_e = sorted(list(set_annids_convVP_loss_e.intersection(set_annids_straight_e)))

    if len(annids_convVP_loss_straight_e) < MIN_MODELS_e:
        print('18) 2 & 7 & 4 : отобранные по сходимости предварительно отобранных по в. п., по loss и по прямоте: недостаточно сетей, продолжение без отбора.....')
        PP_summary.write('энергия: недостаточно сетей, продолжение без отбора, замена на annids_e.....')
        annids_convVP_loss_straight_e = annids_e

    set_annids_convVP_loss_straight_e = set(annids_convVP_loss_straight_e)

    preds_convVP_loss_straight_e = preds_e[preds_e['annid'].isin(annids_convVP_loss_straight_e)]

    # графики предсказаний
    mu_tildeE_convVP_loss_straight, sigma_tildeE_convVP_loss_straight, _, _, \
    mu_E_convVP_loss_straight,      sigma_E_convVP_loss_straight,      _, _ = plotting_combo_func(
            preds_convVP_loss_straight_e, preds_r, maxN_e, maxN_r, n_e, n_r,
            20, variational_min, f'18_convVP_loss_straight',
            df_for_metric_e, df_allN_e, df_for_metric_r, df_allN_r,
            E_exact, rms_exact, E_extrapolation_b,
            plot_limits = plotting_limits, bins = 20)

    PP_summary.write('\n\nЭнергия:\n')    
    PP_summary.write(f'\nsigma_tildeE_convVP_loss_straight = {sigma_tildeE_convVP_loss_straight}\t (S)catter')
    PP_summary.write(f'\nmu_tildeE_convVP_loss_straight = {   mu_tildeE_convVP_loss_straight   }\t "среднее" значение')
    PP_summary.write(f'\nsigma_E_convVP_loss_straight = {     sigma_E_convVP_loss_straight     }\t среднеквадратичное отклонение')
    PP_summary.write(f'\nmu_E_convVP_loss_straight = {        mu_E_convVP_loss_straight        }\t среднее значение')
    PP_summary.write(f"\nannids_convVP_loss_straight_e ({len(annids_convVP_loss_straight_e)}) = \n{annids_convVP_loss_straight_e}")
    PP_summary.write('\n----------------------------------------------------\n')
    #============================================================================================


    #============================================================================================
    PP_summary.write('\n 19) 2 & 7 & 4 & (ns): отобранные по сходимости предварительно отобранных по в. п., по loss, по прямоте и после этого отобранные по n sigma')
    PP_summary.write('\n----------------------------------------------------\n') 

    #отбор по n sigma
    preds_convVP_loss_straight_ns_e, _ = iterative_outlier_filtering_of_predictions(preds_convVP_loss_straight_e, preds_r, method = iterative_outlier_filtering_method)

    annids_convVP_loss_straight_ns_e = sorted(preds_convVP_loss_straight_ns_e['annid'].unique())

    if len(annids_convVP_loss_straight_ns_e) < MIN_MODELS_e:
        print('19) 2 & 7 & 4 & (ns): отобранные по сходимости предварительно отобранных по в. п., по loss, по прямоте и после этого отобранные по n sigma, энергия: недостаточно сетей, продолжение без отбора.....')
        PP_summary.write('энергия: недостаточно сетей, продолжение без отбора, замена на annids_e.....')
        preds_convVP_loss_straight_ns_e =  preds_e
        annids_convVP_loss_straight_ns_e = annids_e

    set_annids_convVP_loss_straight_ns_e = set(annids_convVP_loss_straight_ns_e)

    # графики предсказаний
    mu_tildeE_convVP_loss_straight_ns, sigma_tildeE_convVP_loss_straight_ns, _, _, \
    mu_E_convVP_loss_straight_ns,      sigma_E_convVP_loss_straight_ns,      _, _ = plotting_combo_func(
            preds_convVP_loss_straight_ns_e, preds_r, maxN_e, maxN_r, n_e, n_r,
            20, variational_min, f'19_convVP_loss_straight_ns',
            df_for_metric_e, df_allN_e, df_for_metric_r, df_allN_r,
            E_exact, rms_exact, E_extrapolation_b,
            plot_limits = plotting_limits, bins = 20)

    PP_summary.write('\n\nЭнергия:\n')    
    PP_summary.write(f'\nsigma_tildeE_convVP_loss_straight_ns = {sigma_tildeE_convVP_loss_straight_ns}\t (S)catter')
    PP_summary.write(f'\nmu_tildeE_convVP_loss_straight_ns = {   mu_tildeE_convVP_loss_straight_ns   }\t "среднее" значение')
    PP_summary.write(f'\nsigma_E_convVP_loss_straight_ns = {     sigma_E_convVP_loss_straight_ns     }\t среднеквадратичное отклонение')
    PP_summary.write(f'\nmu_E_convVP_loss_straight_ns = {        mu_E_convVP_loss_straight_ns        }\t среднее значение')
    PP_summary.write(f"\nannids_convVP_loss_straight_ns_e ({len(annids_convVP_loss_straight_ns_e)}) = \n{annids_convVP_loss_straight_ns_e}")
    PP_summary.write('\n----------------------------------------------------\n')    
    #============================================================================================


    #============================================================================================
    PP_summary.write('\n 20) 2 & 3 & 4 & 5: отобранные по loss, в. п., прямоте и сходимости')
    PP_summary.write('\n----------------------------------------------------\n') 
    annids_loss_vp_straight_conv_e = sorted(list(set_annids_loss_e.intersection(set_annids_vp_e, set_annids_straight_e, set_annids_conv_e)))

    if len(annids_loss_vp_straight_conv_e) < MIN_MODELS_e:
        print('20) 2 & 3 & 4 & 5: отобранные по loss, в. п., прямоте и сходимости: энергия: недостаточно сетей, продолжение без отбора.....')
        PP_summary.write('энергия: недостаточно сетей, продолжение без отбора, замена на annids_e.....')
        annids_loss_vp_straight_conv_e = annids_e

    set_annids_loss_vp_straight_conv_e = set(annids_loss_vp_straight_conv_e)

    preds_loss_vp_straight_conv_e = preds_e[preds_e['annid'].isin(annids_loss_vp_straight_conv_e)]

    # графики предсказаний
    mu_tildeE_loss_vp_straight_conv, sigma_tildeE_loss_vp_straight_conv, _, _, \
    mu_E_loss_vp_straight_conv,      sigma_E_loss_vp_straight_conv,      _, _ = plotting_combo_func(
            preds_loss_vp_straight_conv_e, preds_r, maxN_e, maxN_r, n_e, n_r,
            20, variational_min, f'20_loss_vp_straight_conv',
            df_for_metric_e, df_allN_e, df_for_metric_r, df_allN_r,
            E_exact, rms_exact, E_extrapolation_b,
            plot_limits = plotting_limits, bins = 20)

    PP_summary.write('\n\nЭнергия:\n')    
    PP_summary.write(f'\nsigma_tildeE_loss_vp_straight_conv = {sigma_tildeE_loss_vp_straight_conv}\t (S)catter')
    PP_summary.write(f'\nmu_tildeE_loss_vp_straight_conv = {   mu_tildeE_loss_vp_straight_conv   }\t "среднее" значение')
    PP_summary.write(f'\nsigma_E_loss_vp_straight_conv = {     sigma_E_loss_vp_straight_conv     }\t среднеквадратичное отклонение')
    PP_summary.write(f'\nmu_E_loss_vp_straight_conv = {        mu_E_loss_vp_straight_conv        }\t среднее значение')
    PP_summary.write(f"\nannids_loss_vp_straight_conv_e ({len(annids_loss_vp_straight_conv_e)}) = \n{annids_loss_vp_straight_conv_e}")
    PP_summary.write('\n----------------------------------------------------\n')
    #============================================================================================


    #============================================================================================
    PP_summary.write('\n 21) 2 & 3 & 4 & 5 & (ns): отобранные по loss, в. п., прямоте, сходимости, и после этого отобранные по n sigma')
    PP_summary.write('\n----------------------------------------------------\n') 

    #отбор по n sigma
    preds_loss_vp_straight_conv_ns_e, _ = iterative_outlier_filtering_of_predictions(preds_loss_vp_straight_conv_e, preds_r, method = iterative_outlier_filtering_method)

    annids_loss_vp_straight_conv_ns_e = sorted(preds_loss_vp_straight_conv_ns_e['annid'].unique())

    if len(annids_loss_vp_straight_conv_ns_e) < MIN_MODELS_e:
        print('21) 2 & 3 & 4 & 5 & (ns): отобранные по loss, в. п., прямоте, сходимости, и после этого отобранные по n sigma: энергия: недостаточно сетей, продолжение без отбора.....')
        PP_summary.write('энергия: недостаточно сетей, продолжение без отбора, замена на annids_e.....')
        preds_loss_vp_straight_conv_ns_e =  preds_e
        annids_loss_vp_straight_conv_ns_e = annids_e

    set_annids_loss_vp_straight_conv_ns_e = set(annids_loss_vp_straight_conv_ns_e)

    # графики предсказаний
    mu_tildeE_loss_vp_straight_conv_ns, sigma_tildeE_loss_vp_straight_conv_ns, _, _, \
    mu_E_loss_vp_straight_conv_ns,      sigma_E_loss_vp_straight_conv_ns,      _, _ = plotting_combo_func(
            preds_loss_vp_straight_conv_ns_e, preds_r, maxN_e, maxN_r, n_e, n_r,
            20, variational_min, f'21_loss_vp_straight_conv_ns',
            df_for_metric_e, df_allN_e, df_for_metric_r, df_allN_r,
            E_exact, rms_exact, E_extrapolation_b,
            plot_limits = plotting_limits, bins = 20)

    PP_summary.write('\n\nЭнергия:\n')    
    PP_summary.write(f'\nsigma_tildeE_loss_vp_straight_conv_ns = {sigma_tildeE_loss_vp_straight_conv_ns}\t (S)catter')
    PP_summary.write(f'\nmu_tildeE_loss_vp_straight_conv_ns = {   mu_tildeE_loss_vp_straight_conv_ns   }\t "среднее" значение')
    PP_summary.write(f'\nsigma_E_loss_vp_straight_conv_ns = {     sigma_E_loss_vp_straight_conv_ns     }\t среднеквадратичное отклонение')
    PP_summary.write(f'\nmu_E_loss_vp_straight_conv_ns = {        mu_E_loss_vp_straight_conv_ns        }\t среднее значение')
    PP_summary.write(f"\nannids_loss_vp_straight_conv_ns_e ({len( annids_loss_vp_straight_conv_ns_e)}) = \n{annids_loss_vp_straight_conv_ns_e}")
    PP_summary.write('\n----------------------------------------------------\n')    
    #============================================================================================


    #============================================================================================
    PP_summary.write('\n 22) 2 & 4: по loss и по прямоте')
    PP_summary.write('\n Радиус и энергия')
    PP_summary.write('\n----------------------------------------------------\n')   

    annids_loss_straight_e = sorted(list(set_annids_loss_e.intersection(set_annids_straight_e)))
    annids_loss_straight_r = sorted(list(set_annids_loss_r.intersection(set_annids_straight_r)))

    if len(annids_loss_straight_e) < MIN_MODELS_e:
        print('22) 2 & 4: по loss и по прямоте, энергия: недостаточно сетей, продолжение без отбора.....')
        PP_summary.write('энергия: недостаточно сетей, продолжение без отбора, замена на annids_e.....')
        annids_loss_straight_e = annids_e
    
    if len(annids_loss_straight_r) < MIN_MODELS_r:
        print('2) 22) 2 & 4: по loss и по прямоте, радиус: недостаточно сетей, продолжение без отбора.....')
        PP_summary.write('радиус: недостаточно сетей, продолжение без отбора, замена на annids_r.....')
        annids_loss_straight_r = annids_r

    preds_loss_straight_e = preds_e[preds_e['annid'].isin(annids_loss_straight_e)]
    preds_loss_straight_r = preds_r[preds_r['annid'].isin(annids_loss_straight_r)]

    # множества номеров сеток
    set_annids_loss_straight_e = set(annids_loss_straight_e)
    set_annids_loss_straight_r = set(annids_loss_straight_r)

    # графики для предсказаний
    mu_tildeE_loss_straight, sigma_tildeE_loss_straight, mu_tildeRMS_loss_straight, sigma_tildeRMS_loss_straight, \
    mu_E_loss_straight,      sigma_E_loss_straight,      mu_RMS_loss_straight,      sigma_RMS_loss_straight = plotting_combo_func(
            preds_loss_straight_e, preds_loss_straight_r, maxN_e, maxN_r, n_e, n_r,
            20, variational_min, f'22_loss_straight',
            df_for_metric_e, df_allN_e, df_for_metric_r, df_allN_r,
            E_exact, rms_exact, E_extrapolation_b,
            plot_limits = plotting_limits, bins = 20)
    
    PP_summary.write('\n\nЭнергия:\n')    
    PP_summary.write(f'\nsigma_tildeE_loss_straight = {sigma_tildeE_loss_straight}\t (S)catter')
    PP_summary.write(f'\nmu_tildeE_loss_straight = {   mu_tildeE_loss_straight   }\t "среднее" значение')
    PP_summary.write(f'\nsigma_E_loss_straight = {     sigma_E_loss_straight     }\t среднеквадратичное отклонение')
    PP_summary.write(f'\nmu_E_loss_straight = {        mu_E_loss_straight        }\t среднее значение')
    PP_summary.write(f"\nannids_loss_straight_e ({len(annids_loss_straight_e)}) = \n{annids_loss_straight_e}")

    PP_summary.write('\n\nРадиус:\n')    
    PP_summary.write(f'\nsigma_tildeRMS_loss_straight = {sigma_tildeRMS_loss_straight}\t (S)catter')
    PP_summary.write(f'\nmu_tildeRMS_loss_straight = {   mu_tildeRMS_loss_straight   }\t "среднее" значение')
    PP_summary.write(f'\nsigma_RMS_loss_straight = {     sigma_RMS_loss_straight     }\t среднеквадратичное отклонение')
    PP_summary.write(f'\nmu_RMS_loss_straight = {        mu_RMS_loss_straight        }\t среднее значение')   
    PP_summary.write(f"\n\nannids_loss_straight_r ({len(annids_loss_straight_r)}) = \n{annids_loss_straight_r}") 
    PP_summary.write('\n----------------------------------------------------\n')    
    #============================================================================================


    #============================================================================================
    PP_summary.write('\n 23) 2 & 4 & (ns) по loss, по прямоте, и после этого отобранные по n sigma')
    PP_summary.write('\n----------------------------------------------------\n') 

    #отбор по n sigma
    preds_loss_straight_ns_e, preds_loss_straight_ns_r = iterative_outlier_filtering_of_predictions(preds_loss_straight_e, preds_loss_straight_r, method = iterative_outlier_filtering_method)

    annids_loss_straight_ns_e = sorted(preds_loss_straight_ns_e['annid'].unique())
    annids_loss_straight_ns_r = sorted(preds_loss_straight_ns_r['annid'].unique())

    if len(annids_loss_straight_ns_e) < MIN_MODELS_e:
        print('23) 2 & 4 & (ns) по loss, по прямоте, и после этого отобранные по n sigma: энергия: недостаточно сетей, продолжение без отбора.....')
        PP_summary.write('энергия: недостаточно сетей, продолжение без отбора, замена на annids_e.....')
        preds_loss_straight_ns_e  = preds_e
        annids_loss_straight_ns_e = annids_r

    if len(annids_loss_straight_ns_r) < MIN_MODELS_r:
        print('23) 2 & 4 & (ns) по loss, по прямоте, и после этого отобранные по n sigma: энергия: недостаточно сетей, продолжение без отбора.....')
        PP_summary.write('энергия: недостаточно сетей, продолжение без отбора, замена на annids_r.....')
        preds_loss_straight_ns_r  = preds_r
        annids_loss_straight_ns_r = annids_r

    set_annids_loss_straight_ns_e = set(annids_loss_straight_ns_e)
    set_annids_loss_straight_ns_r = set(annids_loss_straight_ns_r)

    # графики предсказаний
    # графики для предсказаний
    mu_tildeE_loss_straight_ns, sigma_tildeE_loss_straight_ns, mu_tildeRMS_loss_straight_ns, sigma_tildeRMS_loss_straight_ns, \
    mu_E_loss_straight_ns,      sigma_E_loss_straight_ns,      mu_RMS_loss_straight_ns,      sigma_RMS_loss_straight_ns = plotting_combo_func(
            preds_loss_straight_ns_e, preds_loss_straight_ns_r, maxN_e, maxN_r, n_e, n_r,
            20, variational_min, f'23_loss_straight_ns',
            df_for_metric_e, df_allN_e, df_for_metric_r, df_allN_r,
            E_exact, rms_exact, E_extrapolation_b,
            plot_limits = plotting_limits, bins = 20)

    PP_summary.write('\n\nЭнергия:\n')    
    PP_summary.write(f'\nsigma_tildeRMS_loss_straight_ns = {sigma_tildeE_loss_straight_ns}\t (S)catter')
    PP_summary.write(f'\nmu_tildeRMS_loss_straight_ns = {   mu_tildeE_loss_straight_ns   }\t "среднее" значение')
    PP_summary.write(f'\nsigma_RMS_loss_straight_ns = {     sigma_E_loss_straight_ns     }\t среднеквадратичное отклонение')
    PP_summary.write(f'\nmu_RMS_loss_straight_ns = {        mu_E_loss_straight_ns        }\t среднее значение')
    PP_summary.write(f"\nannids_loss_straight_ns_r ({len( annids_loss_straight_ns_e)}) = \n{annids_loss_straight_ns_e}")
    PP_summary.write('\n----------------------------------------------------\n')    


    PP_summary.write('\n\nРадиус:\n')    
    PP_summary.write(f'\nsigma_tildeRMS_loss_straight_ns = {sigma_tildeRMS_loss_straight_ns}\t (S)catter')
    PP_summary.write(f'\nmu_tildeRMS_loss_straight_ns = {   mu_tildeRMS_loss_straight_ns   }\t "среднее" значение')
    PP_summary.write(f'\nsigma_RMS_loss_straight_ns = {     sigma_RMS_loss_straight_ns     }\t среднеквадратичное отклонение')
    PP_summary.write(f'\nmu_RMS_loss_straight_ns = {        mu_RMS_loss_straight_ns        }\t среднее значение')
    PP_summary.write(f"\nannids_loss_straight_ns_r ({len( annids_loss_straight_ns_r)}) = \n{annids_loss_straight_ns_r}")
    PP_summary.write('\n----------------------------------------------------\n')        
    #============================================================================================

    # диаграммы перекрытия сеток как множеств индексов 
    # сначала базовые (из пункта I, КРОМЕ ОТОБРАННЫХ ПО Deviation {пушто не используется по сути})
    # энергия
    supervenn_labels = ['все',        'loss',            'vp',             'straight',            'conv',           'convVP']
    supervenn_sets   = [set_annids_e, set_annids_loss_e, set_annids_vp_e,  set_annids_straight_e, set_annids_conv_e, set_annids_convVP_e]
    supervenn(supervenn_sets, supervenn_labels, side_plots=True)
    plt.savefig(pics_path_e / ('intersection_diagram_1-7.' + plot_format), format = plot_format)
    plt.close()
    del supervenn_labels 
    del supervenn_sets


    # затем остальные, почти все (кроме D)
    supervenn_labels = ['1_все',                            '2_loss',                               '3_vp',                             '4_straight',                 '5_conv',                          '7_convVP',\
                        '8_loss_vp',                        '9_loss_vp_straight',                   '10_loss_vp_straight_ns',           '11_vp_conv',                 '12_vp_conv_ns',                   '13_vp_straight_conv',\
                        '14_vp_straight_conv_ns',                                                   '16_convVP_loss',                   '17_convVP_loss_ns',          '18_convVP_loss_straight',         '19_convVP_loss_straight_ns',\
                        '20_loss_vp_straight_conv',         '21_loss_vp_straight_conv_ns',          '22_loss_straight',                 '23_loss_straight_ns']
    supervenn_sets   = [set_annids_e,                       set_annids_loss_e,                      set_annids_vp_e,                    set_annids_straight_e,        set_annids_conv_e,                 set_annids_convVP_e,\
                        set_annids_loss_vp_e,               set_annids_loss_vp_straight_e,          set_annids_loss_vp_straight_ns_e,   set_annids_vp_conv_e,         set_annids_vp_conv_ns_e,           set_annids_vp_straight_conv_e,\
                        set_annids_vp_straight_conv_ns_e,                                           set_annids_convVP_loss_e,           set_annids_convVP_loss_ns_e,  set_annids_convVP_loss_straight_e, set_annids_convVP_loss_straight_ns_e,\
                        set_annids_loss_vp_straight_conv_e, set_annids_loss_vp_straight_conv_ns_e,  set_annids_loss_straight_e,         set_annids_loss_straight_ns_e]

    supervenn(supervenn_sets, supervenn_labels, side_plots=True)
    plt.savefig(pics_path_e / ('intersection_diagram_1-23.' + plot_format), format = plot_format)
    plt.close()
    del supervenn_labels 
    del supervenn_sets


    # радиус
    supervenn_labels = ['все',        'loss',            'straight',            'loss_straight',            'loss_straight_ns']
    supervenn_sets   = [set_annids_r, set_annids_loss_r, set_annids_straight_r, set_annids_loss_straight_r, set_annids_loss_straight_ns_r]
    supervenn(supervenn_sets, supervenn_labels, side_plots=True)
    plt.savefig(pics_path_r / ('intersection_diagram_radius.' + plot_format), format = plot_format)
    plt.close()
    del supervenn_labels 
    del supervenn_sets


    # график kde предсказания - loss
    predictions_loss_kde_plot(preds_e, preds_r, train_loss_e, train_loss_r, '_без_отбора', plotting_limits)

    # 10 для энергии loss_vp_straight_ns
    predictions_loss_kde_plot(preds_loss_vp_straight_ns_e, preds_r, train_loss_e, train_loss_r, '_loss_vp_straight_ns_e', plotting_limits)

    # 19 для энергии
    predictions_loss_kde_plot(preds_convVP_loss_straight_ns_e, preds_r, train_loss_e, train_loss_r, 'convVP_loss_straight_ns_e', plotting_limits)

    # 23 для радиуса и энергии
    predictions_loss_kde_plot(preds_loss_straight_ns_e, preds_loss_straight_ns_r, train_loss_e, train_loss_r, '_loss_straight_ns', plotting_limits)

    
    # гистограммы весов после отбора
    # 10
    weights_loss_vp_straight_ns_e = weights_e[weights_e['annid'].isin(annids_loss_vp_straight_ns_e)]
    weight_histograms(weights_loss_vp_straight_ns_e, path_e, pics_path_e, loss_hist_bins, 'energy_loss_vp_straight_ns_e')
    
    # 19
    weights_convVP_loss_straight_ns_e = weights_e[weights_e['annid'].isin(annids_convVP_loss_straight_ns_e)]
    weight_histograms(weights_convVP_loss_straight_ns_e, path_e, pics_path_e, loss_hist_bins, 'energy_convVP_loss_straight_ns_e')
    
    # 23
    weights_loss_straight_ns_e = weights_e[weights_e['annid'].isin(annids_loss_straight_ns_e)]    
    weight_histograms(weights_loss_straight_ns_e, path_e, pics_path_e, loss_hist_bins, 'energy__loss_straight_ns_e')

    weights_loss_straight_ns_r = weights_r[weights_r['annid'].isin(annids_loss_straight_ns_r)]    
    weight_histograms(weights_loss_straight_ns_r, path_r, pics_path_r, loss_hist_bins, 'radius__loss_straight_ns_e')


    # гистограммы loss для отобранных сетей
    if splitting == True:
        #EN_plus[ EN_plus[ 'hOmega'].isin(common_hO)]
        train_loss_selected_e = train_loss_e[train_loss_e['annid'].isin(set_annids_convVP_loss_straight_ns_e)]
        loss_histograms(train_loss_selected_e, 'loss', path_e, pics_path_e, loss_hist_bins, "energy", reverse_transform_args=rev_tr_args_e, figure_name_addition = '_selected_')
        train_loss_selected_r = train_loss_r[train_loss_r['annid'].isin(set_annids_loss_straight_ns_r)]
        loss_histograms(train_loss_selected_r, 'loss', path_r, pics_path_r, loss_hist_bins, 'radius', reverse_transform_args=rev_tr_args_r, figure_name_addition = '_selected_')
        
        valid_loss_selected_e = valid_loss_e[valid_loss_e['annid'].isin(set_annids_convVP_loss_straight_ns_e)]
        loss_histograms(valid_loss_selected_e, 'val_loss', path_e, pics_path_e, loss_hist_bins, "energy", reverse_transform_args=rev_tr_args_e, figure_name_addition = '_selected_')
        valid_loss_selected_r = valid_loss_r[valid_loss_r['annid'].isin(set_annids_loss_straight_ns_r)]
        loss_histograms(valid_loss_r, 'val_loss', path_r, pics_path_r, loss_hist_bins, 'radius', reverse_transform_args=rev_tr_args_r, figure_name_addition = '_selected_')
        
    if splitting == False:    
        train_loss_selected_e = train_loss_e[train_loss_e['annid'].isin(set_annids_convVP_loss_straight_ns_e)]
        loss_histograms(train_loss_selected_e, 'loss', path_e, pics_path_e, loss_hist_bins, "energy", reverse_transform_args=rev_tr_args_e, figure_name_addition = '_selected_')
        train_loss_selected_r = train_loss_r[train_loss_r['annid'].isin(set_annids_loss_straight_ns_r)]
        loss_histograms(train_loss_selected_r, 'loss', path_r, pics_path_r, loss_hist_bins, 'radius', reverse_transform_args=rev_tr_args_r, figure_name_addition = '_selected_')
    

    

#####
calc_results_df_E, calc_results_df_R = calculate_statistics(preds_e, preds_r, annids_convVP_loss_straight_ns_e, annids_loss_straight_ns_r)

print('results for E: \n') 
print(calc_results_df_E)
PP_summary.write(f'results for E: \n {calc_results_df_E}')
calc_results_df_E.to_excel(path_e / 'calc_results_E.xlsx', index = False, sheet_name = 'E')

print('results for R: \n') 
print(calc_results_df_R)
PP_summary.write(f'results for R: \n {calc_results_df_R}')
calc_results_df_R.to_excel(path_r / 'calc_results_R.xlsx', index = False, sheet_name = 'R')

####

print("PP_done")


PP_end = datetime.datetime.now()
PP_summary.write(f'\n\nPost-processing END: {PP_end}\n')
PP_summary.write(f'\nPost-processing finished in {(PP_end - PP_begin)} or {(PP_end - PP_begin).total_seconds()} second(s). {round(total_num_of_networks * 3600 / float((PP_end - PP_begin).total_seconds()),1)} networks per hour in average')

PP_summary.close()