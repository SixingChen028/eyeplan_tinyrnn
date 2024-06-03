import os 

begin="""#!/bin/bash
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --mem-per-cpu=80G
#SBATCH --time=02:40:00 """ 


for n_rollouts in [2,4,8]:
	job_name="#SBATCH --job-name=tinyRNN_rollout_"+str(n_rollouts)
	output_name="#SBATCH --output tinyRNN_rollout_"+str(n_rollouts)+'.out'
	with open("train_job.sh","w") as f:
		f.write(begin+"\n")
		f.write(job_name+"\n")
		f.write(output_name+"\n")
		f.write('srun -n 1 python -u train.py '+str(n_rollouts)+" 1 \n")
		f.close()
	os.system('sbatch train_job.sh')
                
                
    



