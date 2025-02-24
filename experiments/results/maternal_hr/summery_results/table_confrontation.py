import csv
from collections import defaultdict

def read_and_group_data(file_path):
    grouped_data = defaultdict(list)

    with open(file_path, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            config = f"{row['NeuronType']}, {row['MFs']}, {row['mutation_rate']}, {row['mutation_individual_rate']}, {row['crossover_rate']}"
            grouped_data[config].append({
                "Train_Acc.": float(row["Train_Acc."] or 0),
                "Test_Acc.": float(row["Test_Acc."] or 0),
                "time": float(row["time"] or 0),
            })

    return grouped_data

def read_and_group_data_base(file_path):
    grouped_data = defaultdict(list)

    with open(file_path, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            config = f"{row['NeuronType']}, {row['MFs']}"
            grouped_data[config].append({
                "Train_Acc.": float(row["Train_Acc."] or 0),
                "Test_Acc.": float(row["Test_Acc."] or 0),
                "time": float(row["time"] or 0),
            })

    return grouped_data


def calculate_averages(grouped_data):
    averaged_data = []

    for config, values in grouped_data.items():
        num_entries = len(values)
        avg_train_acc = sum(v["Train_Acc."] for v in values) / num_entries
        avg_test_acc = sum(v["Test_Acc."] for v in values) / num_entries
        avg_time = sum(v["time"] for v in values) / num_entries

        #separete in the configuration the neuron type from the rest
        neuron_type = config.split(",")[0]
        #all the rest of the configuration
        config = config[len(neuron_type):]
        averaged_data.append({
            "Neuron Type": neuron_type[:-1],
            "Configuration": config[2:],
            "Train Acc.": round(avg_train_acc, 3),
            "Test Acc.": round(avg_test_acc, 3),
            "Tempo (s)": round(avg_time, 2),
        })

    return averaged_data


def write_summary_to_csv(output_path, baseline_data, experiment_data):
    header = ["Neuron Type", "Configuration", "Train Acc.", "Test Acc.", "Tempo (s)"]

    with open(output_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()

        writer.writerows(baseline_data)
        writer.writerows(experiment_data)

baseline_file = "experiments/results/maternal_hr/normal_precision/runs_results.csv"
experiment_file = "experiments/results/maternal_hr/precision_calculation/global_result_0.csv"
output_file = "experiments/results/summery_results/summery_results_maternal_micro.csv"

baseline_data = read_and_group_data_base(baseline_file)
baseline_summary = calculate_averages(baseline_data)

experiment_data = read_and_group_data(experiment_file)
experiment_summary = calculate_averages(experiment_data)

write_summary_to_csv(output_file, baseline_summary, experiment_summary)

print(f"Summery into '{output_file}'!")
