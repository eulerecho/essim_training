# Examples

Here we provide a job script to illustrate policy optimization with incrimental learning methods like NPG and PPO. To run the experiments, use the commands below. The experiments are run through the job script provided which tasks two arguments:
- `output`: path to directory where all the results will be saved
- `config`: a config `.txt` file with all the experiment parameters (examples are provided)
The script has to be run from this directory, i.e. `mjrl/examples` 


```
$ python policy_opt_job_script.py --output essim_npg_exp --config example_configs/essim.txt
```


