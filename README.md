This program uses python version 3.8.

There is a requirements.txt file which contains all the packages needed to run this program. Run "pip install -r requirements.txt" to install them.

The program can be run with the line:
"python main.py --n_bandits 5 --distribution bernoulli --time_steps 2000 --repetitions 1000 --experiment_name experiment". 
This line will replicate the results seen in the report (note that you need to run this for both distributions, so with "--distribution gaussian" as well). You can define your own parameters for the remaining given flags.

Additionally you can run the program with the line "python main.py -h" for help on what is expected from each flag.

After the program runs, all plots are saved in the plots folder, which is automatically created in your current directory if it does not exist yet. The experiment name defines the name of file where the plot is saved. "experiment_name_r" is the reward plot, and "experiment_name_p" is the percentage plot. If the same name is used twice the file will be overwritten.

The hyper-parameters defined in the report can be changed in the code itself in file main.py (line 27 to 38).

