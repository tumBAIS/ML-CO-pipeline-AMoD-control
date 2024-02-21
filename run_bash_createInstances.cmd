#!/bin/bash
#SBATCH -J p30w2
#SBATCH -o ./%x.%j.%N.out
#SBATCH -D ./
#SBATCH --get-user-env
#SBATCH --mail-type=end
#SBATCH --mem=20000mb
#SBATCH --cpus-per-task=1
#SBATCH --export=NONE
#SBATCH --time=50:00:00

module load python/3.8.11-base
module load slurm_setup

source venv/bin/activate

echo "We start the generation of training instances"

printf -v HOUR_TWO_DIGIT "%02d" $HOUR
printf -v START_HOUR "%02d" $((HOUR-1))
printf -v END_HOUR "%02d" $((HOUR+1))
TIMEZONE="CET"

DAYS_MONTHS="7,1 8,1 9,1 12,1 13,1"

for DAY_MONTH in ${DAYS_MONTHS}
do
	IFS=',' read DAY MONTH <<< "${DAY_MONTH}"
	YEAR=2015
	printf -v DAY_TWO_DIGIT "%02d" $DAY
	printf -v MONTH_TWO_DIGIT "%02d" $MONTH

	echo ${DAY}
	echo ${MONTH}

	#python create_full_information_solution.py --num_vehicles=${VEHICLES} --start_date="${YEAR}-${MONTH_TWO_DIGIT}-${DAY_TWO_DIGIT} ${START_HOUR}:30:00" --end_date="${YEAR}-${MONTH_TWO_DIGIT}-${DAY_TWO_DIGIT} ${END_HOUR}:30:00" --objective=${OBJECTIVE} --heuristic_distance_range=${DISTANCE_LOCATION} --heuristic_time_range=${HEURISTIC_TIME_RANGE} --sparsity_factor=${SPARSITY}


	echo ${EXTENDED_HORIZON_ARRAY}
	for EXTENDED_HORIZON in ${EXTENDED_HORIZON_ARRAY}
	do
		for STANDARDIZATION in ${STANDARDIZATIONS}
		do
			ITERATE_SPAN=$((3600 / ${NUM_INSTANCES}))
			INSTANCE_DATE="${YEAR}-${MONTH}-${DAY_TWO_DIGIT} ${HOUR_TWO_DIGIT}:00:00"
			ENDE="${YEAR}-${MONTH}-${DAY_TWO_DIGIT} ${END_HOUR}:00:00"

			SECS_BEGIN=$(date +%s --date="$INSTANCE_DATE")
			SECS_IT=$SECS_BEGIN
			SECS_ENDE=$(date +%s --date="$ENDE")

			while [ $SECS_IT -le $SECS_ENDE ]; do
			 	echo instance_date ${INSTANCE_DATE}

				 python create_training_instances.py --num_vehicles=${VEHICLES} --start_date="${YEAR}-${MONTH}-${DAY_TWO_DIGIT} ${START_HOUR}:30:00" --end_date="${YEAR}-${MONTH}-${DAY_TWO_DIGIT} ${END_HOUR}:30:00" --objective=${OBJECTIVE} --instance_date="${INSTANCE_DATE}" --extended_horizon=${EXTENDED_HORIZON} --heuristic_distance_range=${DISTANCE_LOCATION} --heuristic_time_range=${HEURISTIC_TIME_RANGE} --sparsity_factor=${SPARSITY} --policy=$POLICY --rebalancing_action_sampling=0 --standardization=${STANDARDIZATION}

				INSTANCE_DATE=$(date --date="$INSTANCE_DATE ${TIMEZONE} +$ITERATE_SPAN seconds" +"%Y-%m-%d %H:%M:%S")
				SECS_IT=$((SECS_IT+ITERATE_SPAN));
			done
		done
	done
done


