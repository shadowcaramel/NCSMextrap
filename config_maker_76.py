import configparser
import pathlib

cg = configparser.ConfigParser()

# там, где может быть val = None, будет считываться как строка

# некоторые значения вводятся тут, из-за того, что нельзя на них сослаться внутри секции или потому что они они нужны в разных местах
maxN_e = 18
AFC_e  = 1.0
AFB_e  = 0.0
AFC2_e = 1.0

maxN_r = 18
AFC_r  = 1.0
AFB_r  = 0.0
AFC2_r = 1.0


cg['Notes'] = {'Notes': 'blah-blah-blah'}

cg['Script names'] = {
    'training_script_name': 'E_or_R_train_0.42_no_pause',
    'post_processing_script_name': 'PP_script_91'
}


cg['Paths to the data'] = {     # абосолютные пути для таблиц с данными, подразумевается Excel таблицы (обе) с указанием колонок
    'path_energy': '/home/sharypov_re/_данные_для_ML/6He_Daejeon16.xlsx',
    'columns_energy': 'A,B,C',
    'column_weight_energy': 'F',
    'path_radius': '/home/sharypov_re/_данные_для_ML/6He_Daejeon16.xlsx',
    'column_weight_radius': 'F',
    'columns_radius': 'A,B,D'
}


cg['Job parallelism'] = {'script_parts': 4} # количество скриптов-частей


cg['Slurm parameters'] = {
    'job_name': '6He_D16',              # маска имени для расчетов-частей'
    'ntasks': 32,                      # количество запрашиваемых ядер для каждой из задач-частей 
    'PP_ntasks': 8,                   # количество запрашиваемых ядер для пост-процессинга
    'email': "2017104939@pnu.edu.ru", # почта, на которую придет письмо о завершении или об ошибке  
    'nodes': 1                        # количество нод для одной части расчетов
}                 


cg['Train and save parallelism'] = {
    'number_of_threads': cg['Slurm parameters']['ntasks'],  # число создаеваемых процессов при обучении сетей
    'training_threading': True,                            # распараллеливание обучения сетей (а еще сохранения и пр.)
    'training_threading_2': False,                          # "принудительное" разделение на два потока при обучении, если training_threading == True
    'predictions_threading': False                          # многопоточность(!) при предсказаниях (по сути только для загрузки моделей)
}


cg['Data pre-processing logical parameters'] = {    
    'cut_horizontal_E_and_R': False, # обрезание энергии по уровню E_horizontal и СООТВЕТСВУЮЩИХ точек для радиуса
}


cg['Data pre-processing parameters for E'] = {
    'cut_on_min': True,            # обрезание энергии по минимуму
    'horizontal_cut': False,         # горизонтальное обрезание данных для энергии, отрезается всё выше E_horizontal

    'min_Nmax': 2,                   # минимальное модельное пространство, используемое для обучения (или валидации), отбрасываются меньше данного
    'max_Nmax': maxN_e,              # максимальное модельное пространство, используемое для обучения (или валидации)
    'Nmax_for_metric': 2,       # модельное пространство для подсчитывания метрики, характеризующей "хорошесть" обучения; оно не будет входить в тренировочный набор; должно лежать в пределах [minN, maxN]
    
    'hwcutoff_left': 8,              # обрезание данных, которые лежат левее hwcutoff_left по hO
    'hwcutoff_right': 40,           # обрезание данных, которые лежат правее hwcutoff_left по hO
    'hwcutoff_left_for_Extrap_B': 8, # обрезание для энергии по hOmega для вычисления экстраполяции B

    'E_horizontal': 0,               # уровень энергии, по которому обрезаются данные    
    
    'scaler_in_min': 0.0,#'None',#0.0,#'None',         # "None" как вариант; минимум для входных данных (min-max scaler), то есть для Nmax и hO
    'scaler_in_max': 1.0,#'None',#2.0,#'None',         # максимум для входных данных

    'scaler_out_min': 0.0,             # минимум для выходных данных, то есть для энергии
    'scaler_out_max': 1.0,#2.0#(1 + AFB_e) * AFC2_e / 2.0
}


cg['Data pre-processing parameters for R'] = {
    'min_Nmax': 2, 
    'max_Nmax': maxN_r,
    'Nmax_for_metric': 2,
    
    'hwcutoff_left': 12.5,
    'hwcutoff_right': 40,

    

    'scaler_in_min': 0.0,#0.1,#'None',
    'scaler_in_max': 1.0,#0.8,#'None',

    'scaler_out_min': 0.0,#0.1,
    'scaler_out_max': 1.0,#0.8,#(1 + AFB_r) * AFC2_r / 2.0
}


cg['General training parameters'] = {
    'splitting':           False,   # разбивание на тренировочный и валидационный (случайный) наборы
    'bias_usage':          True,    # использование биасов в нейронах
    'Extra_fitting':       False,   # дополнительное обучение в течение нескольких эпох с некоторым batch size и learn rate = LR_e(r)
    'shuffle':             True,    # перемешиваение данных при обучении
    'shuffle_Extra':       False,   # перемешивание при доп. обучении
    'with_sample_weights': False,    # с весом для каждого семпла, вес в таблице с данными
    'write_weights':       True,    # запись весов моделей в файл
}


cg['General numeric training parameters'] = {
    'extra_epochs': 1000,                   # дополнительное обучение в течение нескольких эпох с некоторым batch size и learn rate = LR_e(r)
    'batch_size_extra': 1,                  # BS для extra_fitting

    'LR_extra_coef': 1.0,                   # коэффициент, на который умножается LR в конце CyclicalLearningRate для extra fitting: lr_extra  = lr * lr_extra_coef
    
    'CLR_num_cycles': 80,                   # число циклов в Cyclical Learning Rate (для первой фазы обучения)
    'CLR_coef_maxLR': 100,                  # CLR: LR in [base LR, max LR], max LR = CLR_COEF_MAXLR * base LR
    'CLR_scale_fn_power': 0.75,             # степень огибающей функции вида scale_fn(x) = 1 / x^power для CLR

    'early_stopping_1_loss': 1e-9,          # Callback №1: значение loss, под достижении которого обучение остановится
    
    'early_stopping_2_patience': 500000,     # Callback №2: минимальное количество эпох для отслеживания уменьшения loss в первой фазе обучения
    'early_stopping_2_patience_extra': 400, # Callback №2: минимальное количество эпох для отслеживания уменьшения loss во второй фазе обучения (extra fitting)'

    'number_of_epoch_samples_for_weights_histograms': 100, # количество эпох-семплов, в конце которых записываются веса
}


cg['Numeric training parameters for E'] = {
    'activ_func_coef':   AFC_e,     # activation function = AFC2 * 1 / (1 + exp(-AFC * x)) + AFB
    'activ_func_coef_2': AFC2_e,
    'activ_func_bias':   AFB_e,
    
    'num_of_epochs': 1000000,        # число эпох обучения, общее число эпох = num_of_epochs + epochs_extra
    'batch_size': 'None',          # if BS == None then BS = len(train)
    'learning_rate': 0.0001,       # базовая скорость обучения, она модифицируется CLR
    'learning_rate_decay': 0.98,   # параметр экспоненциального затухания скорости обучения, используется при доп. обучении
    
    'num_of_neural_networks': 256   # количество обучаемых сетей
}


cg['Numeric training parameters for R'] = {
    'activ_func_coef':   AFC_r,
    'activ_func_coef_2': AFC2_r,
    'activ_func_bias':   AFB_r,
    
    'num_of_epochs': 1000000, 
    'batch_size': 'None',     
    'learning_rate': 0.0001, 
    'learning_rate_decay': 0.98, 
    
    'num_of_neural_networks': 256
}


cg['Making predictions parameters'] = {
    'max_Nmax_for_predictions_E': 300,  # максимальный Nmax, для которого делаются предсказания
    'max_Nmax_for_predictions_R': 300,
    'Nmax_step': 2,                     # шаг по Nmax для получения предсказаний   
}




cg['Tensoflow parameters'] = {
    'inter_op_parallelism_threads': 1, # внутренний параллелизм TF
    'floatx': 'float32',               # sets the default float type
    'verbosity': 0                     # информация об обучении в trainsave()
}   


cg['Matplotlib parameters'] = {
    'plot_format':           'jpg',
    'figure_figsize_width':  20.0,          # setting default size of plots
    'figure_figsize_height': 28.0,
    'axes_titlesize':        20,            # fontsize of the axes title
    'axes_labelsize':        20,            # fontsize of the x and y labels
    'xtick_labelsize':       20,            # fontsize of the tick labels
    'ytick_labelsize':       20,
}


cg['General post-processing parameters'] = {
    'Deviation_quantile':                  0.5,    # для отбора по Deviation: вычисляется заданный квантиль и отбрасываются сети, которые дают большее отклонение (для 0.5 - остаются сети, дающее отклонение меньше медианного)
    'plotting_threading':                  False,
    'loss_histograms_bins':                80,     # число разбиений для построения гистограммы распределения loss    
    'predictions_for_nmax_step':           10,     # шаг по Nmax для построения графиков предсказаний для данного Nmax (одного)
    'predictions_for_hOmega_minimal_step': 4,      # минимальный шаг по hOmega для построения графиков предсказаний для данного hOmega
    'loss_selection_quantile':             0.95,   # квантиль для отбора сеток по по loss'у
    
    'iterative_outlier_filtering_method':  'boxplot_rule', # метод итеративного отбора: 'n_sigma' или 'boxplot_rule'
    'n_sigma_selection':                   3,      # сколько сигм брать при итеративном отборе по n sigma
    'boxplot_rule_coef':                   1.5,    # коэффициент в методе boxplot_rule при итеративном отборе
    }

cg['Post-processing parameters for E'] = {       
    'min_number_of_models': 10,               # минимальное количество моделей после проверки вариационного принципа: если остается меньше, то продолжение без отбора  по в.п.
    
    #'exact_value': 'None',                    # точное значение энергии
    #'exact_value': -3.581                    # p_wave
    #'exact_value': -0.150                    # WS08
    #'exact_value': -31.995,                   # 6Li экспериментальное
    #'exact_value': -0.664                    #WS10
    #'exact_value': -2.224,                    # Deutron JISP16
    #'exact_value': 0.8476                    # Res
    #'exact_value': -26.924,                      # 6Be Daejeon16
    'exact_value': -29.269,                     # 6He Daejeon16


    # 'plotting_limits_min': -32.1,
    # 'plotting_limits_max': -31.7, # Li

    #'plotting_limits_min': -3.7,
    #'plotting_limits_max': -3.5, # pwave

    #'plotting_limits_min': -1.0,
    #'plotting_limits_max': -1.0, # swave

    # 'plotting_limits_min': -2.5,
    # 'plotting_limits_max': -2.0, # Deutron JISP16

    # 'plotting_limits_min': -2.30,
    # 'plotting_limits_max': -1.90, # Deutron JISP16

    # 'plotting_limits_min': -27.5,
    # 'plotting_limits_max': -25.5, # 6Be Daejeon16

    'plotting_limits_min': -29.5,
    'plotting_limits_max': -29.3, # 6Be Daejeon16

    # отбор по исходимости:
    'eps_ho':                     0.02,             # критерий сходимости для разницы по разным hOmega: E_min(hO) - E_max(hO)
    'eps_nmax':                   0.001,            # критерий сходимости для разницы по разным Nmax:   E_min(Nmax_i) - E_min(Nmax_j)
    'convergence_left_quantile':  0.0,              # левый  квантиль для отбора по сходимости
    'convergence_right_quantile': 0.8,             # правый квантиль для отбора по сходимости

    # вариационный принцип:
    'VP_check_mode': 2, # способ проверки вариационного принципа:
                        # 1: первое сравнение - сравнение предсказаний следующего Nmax с исходными данными
                        # 2: первое сравнение - сравнение предсказаний следующего Nmax с предсказаниями
    'max_Nmax_for_variational_principle': 300, # Nmax, до которого проверятется вариационный принцип
    'VP_hO_left':  cg['Data pre-processing parameters for E']['hwcutoff_left'],  # левая граница по hOmega при проверке вариационного принципа (может отличаться от входных данных)
    'VP_hO_right': cg['Data pre-processing parameters for E']['hwcutoff_right'], # правая граница по hOmega при проверке вариационного принципа
    'VP_hO_minimal_step':  2.0,  # минимальный шаг по hO: если шаг в предсказаниях меньше минимального, то используется заданный минимальный шаг, а не тот, который в предсказаниях
    'VP_epsilon':          1e-4, # допустимая разница при сравнении энергии при соседних Nmax при данном hOmega, т. е. допустимое нарушение вариационного принципа "локально"
    'VP_additional_check': False, # дополнительная проверка в. п.: сравниваются мод. пр-ва Nmax = max_Nmax_for_variational_principle и Nmax = max_Nmax_for_predictions_E без промежуточных
    'VP_epsilon_2':        1e-3,  # допустимое нарушение в. п. при дополнительной проверке
    
    'straightness_tolerance': 0.002, # проверка на "прямость линий": параметр допуска на "непрямоту" (макс. разница между минимумом и максимумом)

}

cg['Post-processing parameters for R'] = {    
    'min_number_of_models': 10, # минимальное количество моделей после проверки по производной: если остается меньше, то продолжение без отбора    
    
    'exact_value': 'None',      # точное значение радиуса
    #'exact_value': 3.450       # pwave
    #'exact_value': 10.695      # WS08    
    #'exact_value': 5.893       # WS10 s_1/2-wave, HC
    #'exact_value': 1.9643,      # Deutron JISP16
    #'exact_value': -26.924,      # 6Be Daejeon16

    # 'plotting_limits_min': 2.2,
    # 'plotting_limits_max': 2.6, # Li

    # 'plotting_limits_min': 2.0,
    # 'plotting_limits_max': 3.0, # 6Be Daejeon16

    #'plotting_limits_min': 3.2,
    #'plotting_limits_max': 3.6, # pwave

    #'plotting_limits_min': 4.0,
    #'plotting_limits_max': 8.0, # swave

    # 'plotting_limits_min': 1.5,
    # 'plotting_limits_max': 2.2, # Deutron JISP16

    # 'plotting_limits_min': 1.7,
    # 'plotting_limits_max': 2.2, # Deutron JISP16

    'plotting_limits_min': 1.8,
    'plotting_limits_max': 2.0, # 6He Daejeon16

    'straightness_tolerance': 0.005, # проверка на "прямость линий": параметр допуска на "непрямоту" (макс. разница между минимумом и максимумом)
}


with open(pathlib.Path(__file__).resolve().parent / 'config.ini', 'w') as configfile:
    cg.write(configfile)