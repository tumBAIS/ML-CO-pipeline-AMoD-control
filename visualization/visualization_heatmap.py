import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from src import config, util
import os
from prep.preprocess_grid import RebalancingGrid


def get_rebalancing_grid(args):
    return RebalancingGrid(args=args)


def get_vehicle_distribution(vehicleList, rebalancing_grid):
    vehicle_heatmap = np.zeros(args.amount_of_cellsRebalancing[0] * args.amount_of_cellsRebalancing[1])

    for vehicle in vehicleList:
        idx_vehicle = rebalancing_grid.get_original_idx(longitude=vehicle.get("location_lon"), latitude=vehicle.get("location_lat"))
        vehicle_heatmap[idx_vehicle] = vehicle_heatmap[idx_vehicle] + 1

    # incorporate nan
    vehicle_heatmap = np.array([i if i > 0 else np.nan for i in vehicle_heatmap])
    vehicle_heatmap = vehicle_heatmap.reshape(args.amount_of_cellsRebalancing[0], args.amount_of_cellsRebalancing[1])
    vehicle_heatmap = pd.DataFrame(vehicle_heatmap, columns=np.arange(args.lon_lat_area[0], args.lon_lat_area[1], (args.lon_lat_area[1]-args.lon_lat_area[0])/args.amount_of_cellsRebalancing[0]),
                                   index=np.arange(args.lon_lat_area[2], args.lon_lat_area[3], (args.lon_lat_area[3]-args.lon_lat_area[2])/args.amount_of_cellsRebalancing[1]))
    return vehicle_heatmap


if __name__ == '__main__':
    os.chdir('../')
    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "font.family": "serif",
    }

    plt.rcParams.update(tex_fonts)

    colormap_name = "YlOrRd"
    cmap = matplotlib.cm.get_cmap(colormap_name)

    args = config.parser.parse_args()

    directory_offline = (args.directory_save_data_visualization + "Source_obj-profit_offline_startDat-2015-01-26 09:00:00_endDat-2015-01-26 10:00:00_fleetSi-500/", "FI")
    directory_SB = (args.directory_save_data_visualization + "Source_obj-profit_policy_SB_startDat-2015-01-26 09:00:00_endDat-2015-01-26 10:00:00_fleetSi-500/", "SB")
    directory_CB = (args.directory_save_data_visualization + "Source_obj-profit_policy_CB_startDat-2015-01-26 09:00:00_endDat-2015-01-26 10:00:00_fleetSi-500_cap-10/", "CB")
    directory_sampling = (args.directory_save_data_visualization + "Source_obj-profit_sampling_startDat-2015-01-26 09:00:00_endDat-2015-01-26 10:00:00_fleetSi-500/", "sampling")

    list_of_times = ["2015-01-26 09:00:00", "2015-01-26 09:20:00", "2015-01-26 09:40:00", "2015-01-26 09:50:00"]
    list_of_metrics = ["vehicle_distribution"]
    #list_of_solution_directories = [directory_offline, directory_sampling]#, directory_SLPA, directory_SLCA]
    list_of_solution_directories = [directory_SB, directory_CB]
    rebalancing_grid = get_rebalancing_grid(args)
    v_max_dict = {"vehicle_distribution": 11}
    manhattan_shape = util.get_manhattan_exact_shapefile(args)
    for metric_idx, metric in enumerate(list_of_metrics):
        vmax_metric = 0
        fig, axs = plt.subplots(len(list_of_solution_directories), len(list_of_times), figsize=(6.2, len(list_of_solution_directories) * 3))
        axs = axs.flatten()
        for solution_directory_idx, (solution_directory, solution_name) in enumerate(list_of_solution_directories):
            for solution_file in os.listdir(solution_directory):
                if "Source" not in solution_file:
                    continue
                time = solution_file.split("now:")[-1]
                if time not in list_of_times:
                    continue
                time_idx = list_of_times.index(time)

                instance_input_file = open(solution_directory + solution_file, "rb")
                instance_dict = pickle.load(instance_input_file)
                instance_input_file.close()

                solution_edges = instance_dict["solution_edges"]
                requestList = instance_dict["requestList"]
                vehicleList = instance_dict["vehicleList"]
                vehicleRoutes = instance_dict["vehicleRoutes"]
                verticesList = instance_dict["verticesList"]

                metric_heatmap = get_vehicle_distribution(vehicleList, rebalancing_grid)

                axs_index = solution_directory_idx * len(list_of_times) + time_idx
                if np.nanmax(metric_heatmap.values) > vmax_metric:
                    vmax_metric = np.nanmax(metric_heatmap.values)
                manhattan_shape.boundary.plot(facecolor=cmap(0), edgecolor="black", ax=axs[axs_index], zorder=1)  # "none"
                pc = axs[axs_index].pcolormesh(list(metric_heatmap.columns), list(metric_heatmap.index), metric_heatmap.values, cmap=colormap_name, zorder=2, vmin=0, vmax=v_max_dict[metric])
                if solution_directory_idx == 0:
                    axs[axs_index].title.set_text(time.split(" ")[1])
                if time_idx == 0:
                    axs[axs_index].set_ylabel(solution_name)
                axs[axs_index].axis(xmin=-74.03, xmax=-73.9, ymin=40.678, ymax=40.888)
                axs[axs_index].set_xticks([])
                axs[axs_index].set_yticks([])
        for ax in axs:
            ax.tick_params(color='lightgrey')
            for spine in ax.spines.values():
                spine.set_edgecolor('lightgrey')
        fig.subplots_adjust(right=0.8, wspace=0, hspace=0)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.6])  # add
        cbar_ax.set_label('num. of vehicles in rebalancing cell')  # add
        fig.colorbar(pc, cax=cbar_ax, label='num. of vehicles in rebalancing cell')  # add
        plt.savefig(args.directory_save_data_visualization + "{}2.pdf".format(metric), bbox_inches="tight")
        plt.show()
        print("The vmax of {} is {}, the correct value would be {}".format(metric, v_max_dict[metric], vmax_metric))