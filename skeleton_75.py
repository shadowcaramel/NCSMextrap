import pathlib
import shutil
import configparser
import os
import subprocess
import re # regular expressions
import datetime
import runpy

config_file_name = 'config_maker_76.py'

folder_name_addition = '6He_PR_maxNmax18' # скобки в путях и именах файлов не допускаются

now = datetime.datetime.now() # current date and time

year        = now.strftime("%Y")
month       = now.strftime("%m")
day         = now.strftime("%d")
hour_minute = now.strftime("%H-%M")

path_addition = day + "_" + month + "_" + year + '_' + hour_minute + '_' + folder_name_addition # для имени папки

working_folder = pathlib.Path.cwd() / path_addition

working_folder.mkdir(parents=True, exist_ok=True)                       # создание рабочей папки
shutil.copyfile(config_file_name, working_folder / config_file_name)  # копирование туда config_maker.py для последующего выполнения


# запуск скрипта, создающего конфиг-файл
#print(f'working folder = {working_folder}')
runpy.run_path(path_name = working_folder / config_file_name)

# чтение конфига
config = configparser.ConfigParser()
config.read(working_folder / 'config.ini')

summary = open(working_folder / 'skeleton_summary.txt', 'w', encoding = 'utf8') # текстовый файл с различными сведениями

# считываем из конфигурационного файла нужные для распараллеливания величины
parallelism = config['Job parallelism']

SCRIPT_PARTS = int(parallelism['script_parts']) # количество скриптов-частей

MAIN_SCRIPT = config['Script names']['training_script_name']
PP_SCRIPT   = config['Script names']['post_processing_script_name']

Slurm_params = config['Slurm parameters']
JOB_NAME = Slurm_params['job_name']
NTASKS   = int(Slurm_params['ntasks'])
PP_NTASKS   = int(Slurm_params['pp_ntasks'])
EMAIL    = Slurm_params['email']
NODES    = Slurm_params['nodes']


partial_job_ids = [] # jobID для расчетов-частей
 

for script_part in range(SCRIPT_PARTS):
    path_to_part = pathlib.Path(working_folder / f'part_{script_part}') # путь к папкам для частей    
    path_to_part.mkdir(parents=True, exist_ok=True) # создание папок для частей

    # копирование основного скрипта в папки для частей для последующего выполнения
    path_to_main_script_part = path_to_part / f"{MAIN_SCRIPT}_part_{script_part}.py"
    shutil.copy(pathlib.Path.cwd() / f"{MAIN_SCRIPT}.py", path_to_main_script_part)

    #создание job-файла
    with open(path_to_part / 'partial_job.sh', 'w') as jf:
        jf.writelines("#!/bin/bash\n")
        jf.write(f"#SBATCH -J {JOB_NAME}_p{script_part}        # --job-name\n")
        jf.write(f"#SBATCH -o {path_to_part}/job.%j.out 	   # --output (Name of stdout output file, %j - jobId)\n")
        jf.write(f"#SBATCH -N {NODES}            	           # --nodes (Number of nodes)\n")
        jf.write(f"#SBATCH -n {NTASKS}           	           # --ntasks (Number of MPI tasks)\n")
        jf.write("#SBATCH --mail-type=END,FAIL\n")
        jf.write(f"#SBATCH --mail-user={EMAIL}\n")
        jf.write("#SBATCH --threads-per-core=1\n")
        jf.write("date; hostname; pwd\n")

        jf.write("source ~/.bashrc\n")
        jf.write("#module load miniconda3\n")
        jf.write("conda activate tf_w_addons\n")

        jf.write(f"cd {path_to_part}\n")
        jf.write(f"python {path_to_main_script_part}\n")

    command_string = f"sbatch {path_to_part / 'partial_job.sh'}" # команда на выполнение job'a
    #subprocess.run(command_string, capture_output=True) 
    #proc = subprocess.Popen([command_string], shell=True, capture_output = True) # посылает команду на выполнение, принимает выход
    proc = subprocess.run([command_string], shell=True, capture_output = True)

    #берем выходную строку, извлекаем из нее JobID
    output_string = proc.stdout.decode()    
    job_id = re.findall('[0-9]+', output_string)[0]    
    partial_job_ids.append(int(job_id))

print(partial_job_ids)


# создаем папку пост-процессинга, копируем туда соотв. скрипт и запускаем его
path_to_PP = pathlib.Path(working_folder /'PP') # путь к папкам для частей    
path_to_PP.mkdir(parents=True, exist_ok=True) # создание папок для частей
path_to_PP_script = path_to_PP / f"{PP_SCRIPT}.py"
shutil.copy(pathlib.Path.cwd() / f"{PP_SCRIPT}.py", path_to_PP_script)

# создаем job-file пост-процессинга
with open(path_to_PP / 'PP_job.sh', 'w') as jf:
        jf.writelines("#!/bin/bash\n")
        jf.write(f"#SBATCH -J {JOB_NAME}_PP             # --job-name\n")
        jf.write(f"#SBATCH -o {path_to_PP}/job.%j.out   # --output (Name of stdout output file, %j - jobId)\n")
        jf.write(f"#SBATCH -N {NODES}            	    # --nodes (Number of nodes)\n")
        jf.write(f"#SBATCH -n {PP_NTASKS}           	# --ntasks (Number of MPI tasks)\n")
        jf.write("#SBATCH --mail-type=END,FAIL\n")
        jf.write(f"#SBATCH --mail-user={EMAIL}\n")
        jf.write("#SBATCH --threads-per-core=1\n")
        jf.write("date; hostname; pwd\n")

        jf.write("source ~/.bashrc\n")
        jf.write("#module load miniconda3\n")
        jf.write("conda activate tf_w_addons\n")

        jf.write(f"cd {path_to_PP}\n")
        jf.write(f"python {path_to_PP_script}\n")

dependency_string = f"--dependency=afterok:{','.join(map(str, partial_job_ids))}" # список в строку без скобок и пробелов
#print(dependency_string)
command_string = f"sbatch {dependency_string} {path_to_PP / 'PP_job.sh'}" # команда на выполнение job'a с зависимостью
#print(command_string)
#proc = subprocess.Popen([command_string], shell=True, capture_output = True) # посылает команду на выполнение, принимает выход
proc = subprocess.run([command_string], shell=True, capture_output = True)

output_string = proc.stdout.decode()
PP_job_id = re.findall('[0-9]+', output_string)[0]
print(PP_job_id)



summary.write(f'skeleton begin: {str(now)}')
summary.write(f'\nPartial job IDs = {partial_job_ids}')
summary.write(f'\nPost-processing JobID = {PP_job_id}')
summary.close()