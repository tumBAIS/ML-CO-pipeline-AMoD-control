#!/bin/bash

running_type=vehicleTest #hyperparameterSelection #hyperparameterOffline_distance #hyperparameterOffline_time #sparsityTest #vehicleTest #samplingHyperparameterSelection #samplingReduceFutureSelect #testmaxcapacity hyperparameterDeepLearning #sanity_check

# Default settings
OBJECTIVE="profit"
HEURISTIC_TIME_RANGE=720
DISTANCE_LOCATION=1.5
DISTANCE_LOCATION_ARRAY_HYP=$DISTANCE_LOCATION
HEURISTIC_TIME_RANGE_ARRAY_HYP=$HEURISTIC_TIME_RANGE
VEHICLE_ARRAY=2000
SPARSITY_ARRAY=0.3
BENCHMARK_POLICIES="offline greedy sampling"
MAXIMUM_CAPACITY_ARRAY_HYP="1"
REDUCE_WEIGHT_OF_FUTURE_REQUEST_FACTOR_ARRAY="1.0"
HOUR=9
NUM_INSTANCES="15"
NUM_PERTURBATIONS=50
EXTENDED_HORIZON_ARRAY=5
MODELS="Linear"
LEARNING_RATE_ARRAY="0.01"
LAYER_SIZE_ARRAY=10
NUMLAYERS_ARRAY=2
NORMALIZATIONS=0
SANITY=0
SANITY_PERFORMANCE=0


if [ ${running_type} == hyperparameterSelection ]; then
	EXTENDED_HORIZON_ARRAY="5 10 15 20 25 30 35 40"
	ABBREVIATION="hS"
	POLICIES="policy_CB policy_SB"
elif [ ${running_type} == hyperparameterDeepLearning ]; then
	VEHICLE_ARRAY="500 1000 2000 3000 4000 5000"
	LEARNING_RATE_ARRAY="0.1 0.001"
	LAYER_SIZE_ARRAY="10 1000"
	NUMLAYERS_ARRAY="0 3"
	STANDARDIZATIONS="100 0"
	ABBREVIATION="hL"
	POLICIES="policy_CB policy_SB"
	MODELS="NN"
elif [ ${running_type} == hyperparameterOffline_distance ]; then
	DISTANCE_LOCATION_ARRAY_HYP="0.3 0.5 0.7 0.9 1.1 1.3 1.5 1.7"
	HEURISTIC_TIME_RANGE_ARRAY_HYP="100000000"
	BENCHMARK_POLICIES="offline"
	SPARSITY_ARRAY="0.3"
	ABBREVIATION="hO_d"
elif [ ${running_type} == hyperparameterOffline_time ]; then
	DISTANCE_LOCATION_ARRAY_HYP="200"
	HEURISTIC_TIME_RANGE_ARRAY_HYP="120 180 360 540 720 900"
	BENCHMARK_POLICIES="offline"
	SPARSITY_ARRAY="0.3"
	ABBREVIATION="hO_t"
elif [ ${running_type} == samplingHyperparameterSelection ]; then
	EXTENDED_HORIZON_ARRAY="5 10 15 20 25 30 35 40"
	BENCHMARK_POLICIES="sampling"
	ABBREVIATION="sHS"
elif [ ${running_type} == samplingReduceFutureSelect ]; then
	REDUCE_WEIGHT_OF_FUTURE_REQUEST_FACTOR_ARRAY="1.0 0.95 0.9 0.85 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1"
	BENCHMARK_POLICIES="sampling"
	ABBREVIATION="rfs"
elif [ ${running_type} == sparsityTest ]; then
	SPARSITY_ARRAY="0.1 0.2 0.3 0.4 0.5 0.6 0.7 1.0"
	ABBREVIATION="sT"
	POLICIES="policy_CB policy_SB"
elif [ ${running_type} == vehicleTest ]; then
	VEHICLE_ARRAY="500 1000 2000 3000 4000 5000"
	ABBREVIATION="vT"
	POLICIES="policy_CB policy_SB"
	MODELS="NN Linear"
elif [ ${running_type} == sanity_check ]; then
	VEHICLE_ARRAY="500 1000 2000 3000 4000 5000"
	ABBREVIATION="sc"
	POLICIES="policy_CB policy_SB"
	MODELS="NN Linear"
	SANITY=1
elif [ ${running_type} == sanity_check_performance ]; then
	VEHICLE_ARRAY="500 1000 2000 3000 4000 5000"
	ABBREVIATION="sc"
	POLICIES="policy_CB policy_SB"
	MODELS="NN Linear"
	SANITY_PERFORMANCE=1
elif [ ${running_type} == testmaxcapacity ]; then
	MAXIMUM_CAPACITY_ARRAY_HYP="1 2 3 4 5 6 7 8 9 10 11 12"
	POLICIES="policy_CB"
	ABBREVIATION="tmC"
else
	echo "Wrong running type specified"
fi


for SPARSITY in ${SPARSITY_ARRAY}
do
	for VEHICLES in ${VEHICLE_ARRAY}
	do
		for POLICY in ${POLICIES}
		do
			if [[ ${POLICY} == "policy_CB" ]]; then
				if [[ ${running_type} != hyperparameterDeepLearning ]]; then
					STANDARDIZATIONS=100
				fi
				if [[ ${running_type} != hyperparameterSelection && ${running_type} != samplingHyperparameterSelection ]]; then
					EXTENDED_HORIZON_ARRAY=5
				fi
			elif [[ ${POLICY} == "policy_SB" ]]; then
				if [[ ${running_type} != hyperparameterDeepLearning ]]; then
					STANDARDIZATIONS=0
				fi
				if [[ ${running_type} != hyperparameterSelection && ${running_type} != samplingHyperparameterSelection ]]; then
					EXTENDED_HORIZON_ARRAY=5
				fi
			fi
			echo "run generation of training instances"
			#sbatch --job-name=i$ABBREVIATION$SPARSITY$VEHICLES --export=VEHICLES=$VEHICLES,OBJECTIVE=$OBJECTIVE,EXTENDED_HORIZON_ARRAY="${EXTENDED_HORIZON_ARRAY}",SPARSITY=$SPARSITY,HOUR=$HOUR,HEURISTIC_TIME_RANGE=$HEURISTIC_TIME_RANGE,DISTANCE_LOCATION=$DISTANCE_LOCATION,POLICY=$POLICY,NUM_INSTANCES=$NUM_INSTANCES,STANDARDIZATIONS="${STANDARDIZATIONS}" run_bash_createInstances.cmd
			sleep 0.01
			for EXTENDED_HORIZON in ${EXTENDED_HORIZON_ARRAY}
			do
				for MODEL in ${MODELS}
				do
					if [[ ${running_type} != hyperparameterDeepLearning && ${MODEL} == "NN" ]]; then
						if [[ ${POLICY} == "policy_CB" && ${VEHICLES} == 500 ]]; then
							LEARNING_RATE_ARRAY="0.1"
							LAYER_SIZE_ARRAY="10.0"
							STANDARDIZATIONS="100"
							NUMLAYERS_ARRAY="0"
						elif [[ ${POLICY} == "policy_CB" && ${VEHICLES} == 1000 ]]; then
							LEARNING_RATE_ARRAY="0.1"
							LAYER_SIZE_ARRAY="1000.0"
							STANDARDIZATIONS="100"
							NUMLAYERS_ARRAY="3"
						elif [[ ${POLICY} == "policy_CB" && ${VEHICLES} == 2000 ]]; then
							LEARNING_RATE_ARRAY="0.1"
							LAYER_SIZE_ARRAY="10.0"
							STANDARDIZATIONS="0"
							NUMLAYERS_ARRAY="0"
						elif [[ ${POLICY} == "policy_CB" && ${VEHICLES} == 3000 ]]; then
							LEARNING_RATE_ARRAY="0.1"
							LAYER_SIZE_ARRAY="10.0"
							STANDARDIZATIONS="100"
							NUMLAYERS_ARRAY="0"
						elif [[ ${POLICY} == "policy_CB" && ${VEHICLES} == 4000 ]]; then
							LEARNING_RATE_ARRAY="0.1"
							LAYER_SIZE_ARRAY="1000.0"
							STANDARDIZATIONS="0"
							NUMLAYERS_ARRAY="3"
						elif [[ ${POLICY} == "policy_CB" && ${VEHICLES} == 5000 ]]; then
							LEARNING_RATE_ARRAY="0.1"
							LAYER_SIZE_ARRAY="1000.0"
							STANDARDIZATIONS="100"
							NUMLAYERS_ARRAY="3"
						elif [[ ${POLICY} == "policy_SB" && ${VEHICLES} == 500 ]]; then
							LEARNING_RATE_ARRAY="0.001"
							LAYER_SIZE_ARRAY="10.0"
							STANDARDIZATIONS="100"
							NUMLAYERS_ARRAY="3"
						elif [[ ${POLICY} == "policy_SB" && ${VEHICLES} == 1000 ]]; then
							LEARNING_RATE_ARRAY="0.001"
							LAYER_SIZE_ARRAY="1000.0"
							STANDARDIZATIONS="100"
							NUMLAYERS_ARRAY="3"
						elif [[ ${POLICY} == "policy_SB" && ${VEHICLES} == 2000 ]]; then
							LEARNING_RATE_ARRAY="0.001"
							LAYER_SIZE_ARRAY="1000.0"
							STANDARDIZATIONS="100"
							NUMLAYERS_ARRAY="3"
						elif [[ ${POLICY} == "policy_SB" && ${VEHICLES} == 3000 ]]; then
							LEARNING_RATE_ARRAY="0.001"
							LAYER_SIZE_ARRAY="1000.0"
							STANDARDIZATIONS="0"
							NUMLAYERS_ARRAY="3"
						elif [[ ${POLICY} == "policy_SB" && ${VEHICLES} == 4000 ]]; then
							LEARNING_RATE_ARRAY="0.1"
							LAYER_SIZE_ARRAY="1000.0"
							STANDARDIZATIONS="0"
							NUMLAYERS_ARRAY="3"
						elif [[ ${POLICY} == "policy_SB"&& ${VEHICLES} == 5000 ]]; then
							LEARNING_RATE_ARRAY="0.001"
							LAYER_SIZE_ARRAY="1000.0"
							STANDARDIZATIONS="0"
							NUMLAYERS_ARRAY="0"
						fi
					elif [[ ${POLICY} == "policy_CB" && ${MODEL} == "Linear" ]]; then
						STANDARDIZATIONS=100
					elif [[ ${POLICY} == "policy_SB" && ${MODEL} == "Linear" ]]; then
						STANDARDIZATIONS=0
					fi

					for LEARNING_RATE in ${LEARNING_RATE_ARRAY}
					do
						for LAYER_SIZE in ${LAYER_SIZE_ARRAY}
						do
							for NUMLAYERS in ${NUMLAYERS_ARRAY}
							do
								for STANDARDIZATION in ${STANDARDIZATIONS}
								do
									for NORMALIZATION in ${NORMALIZATIONS}
									do
										echo "run training"
										echo $POLICY $MODEL $STANDARDIZATION
										sbatch --job-name=l$ABBREVIATION$MODEL$SPARSITY$VEHICLES --export=OBJECTIVE=$OBJECTIVE,EXTENDED_HORIZON=$EXTENDED_HORIZON,VEHICLES=$VEHICLES,SPARSITY=$SPARSITY,HEURISTIC_TIME_RANGE_ARRAY_HYP="${HEURISTIC_TIME_RANGE_ARRAY_HYP}",DISTANCE_LOCATION_ARRAY_HYP="${DISTANCE_LOCATION_ARRAY_HYP}",MAXIMUM_CAPACITY_ARRAY_HYP="${MAXIMUM_CAPACITY_ARRAY_HYP}",STANDARDIZATION=$STANDARDIZATION,POLICY=$POLICY,HEURISTIC_TIME_RANGE=$HEURISTIC_TIME_RANGE,DISTANCE_LOCATION=$DISTANCE_LOCATION,HOUR=$HOUR,RUNNING_TYPE=$running_type,NUM_PERTURBATIONS=$NUM_PERTURBATIONS,MODEL=$MODEL,LEARNING_RATE=$LEARNING_RATE,LAYER_SIZE=$LAYER_SIZE,NORMALIZATION=$NORMALIZATION,NUMLAYERS=$NUMLAYERS,SANITY=$SANITY,SANITY_PERFORMANCE=$SANITY_PERFORMANCE run_bash_training.cmd
										sleep 0.01
									done
								done
							done
						done
					done
				done
			done
		done
		for BENCHMARK_POLICY in ${BENCHMARK_POLICIES}
		do
			for DISTANCE_LOCATION_HYP in ${DISTANCE_LOCATION_ARRAY_HYP}
			do
				for HEURISTIC_TIME_RANGE_HYP in ${HEURISTIC_TIME_RANGE_ARRAY_HYP}
				do
					for EXTENDED_HORIZON in ${EXTENDED_HORIZON_ARRAY}
					do
						for REDUCE_WEIGHT_OF_FUTURE_REQUEST_FACTOR in ${REDUCE_WEIGHT_OF_FUTURE_REQUEST_FACTOR_ARRAY}
						do
							echo "run evaluation"
							#sbatch --job-name=b$ABBREVIATION --export=EXTENDED_HORIZON=$EXTENDED_HORIZON,VEHICLES=$VEHICLES,OBJECTIVE=$OBJECTIVE,DISTANCE_LOCATION_HYP=$DISTANCE_LOCATION_HYP,HEURISTIC_TIME_RANGE_HYP=$HEURISTIC_TIME_RANGE_HYP,SPARSITY=$SPARSITY,REDUCE_WEIGHT_OF_FUTURE_REQUEST_FACTOR=$REDUCE_WEIGHT_OF_FUTURE_REQUEST_FACTOR,POLICY=$BENCHMARK_POLICY,RUNNING_TYPE=$running_type,HOUR=$HOUR run_bash_benchmarks.cmd
							sleep 0.01
						done
					done
				done
			done
		done
	done
done
