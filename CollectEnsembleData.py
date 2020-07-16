from CollectUtils import *

np.random.seed(42)

base_path_list = ["HeatH1_50b", "HeatH1_100b", "HeatH1_200b",
                  "WaveH1_30", "WaveH1_60", "WaveH1_90", "WaveH1_120",
                  "WaveH1_30b", "WaveH1_60b", "WaveH1_90b", "WaveH1_120b",
                  "StokesH1_20", "StokesH1_40", "StokesH1_80", "StokesH1_160"]

# base_path_list = ["HeatH1_50b", "HeatH1_100b", "HeatH1_200b"]

for base_path in base_path_list:
    print("#################################################")
    print(base_path)

    b = False
    compute_std = False
    directories_model = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    sensitivity_df = pd.DataFrame(columns=["batch_size",
                                           "regularization_parameter",
                                           "kernel_regularizer",
                                           "neurons",
                                           "hidden_layers",
                                           "residual_parameter",
                                           "L2_norm_test",
                                           "error_train",
                                           "error_val",
                                           "error_test"])
    # print(sensitivity_df)
    selection_criterion = "error_train"

    Nu_list = []
    Nf_list = []
    t_0 = 0
    t_f = 1
    x_0 = -1
    x_f = 1

    L2_norm = []
    criterion = []
    best_retrain_list = []
    list_models_setup = list()

    for subdirec in directories_model:
        model_path = base_path

        sample_path = model_path + "/" + subdirec
        retrainings_fold = [d for d in os.listdir(sample_path) if os.path.isdir(os.path.join(sample_path, d))]

        retr_to_check_file = None
        for ret in retrainings_fold:
            if os.path.isfile(sample_path + "/" + ret + "/TrainedModel/Information.csv"):
                retr_to_check_file = ret
                break

        setup_num = int(subdirec.split("_")[1])
        if retr_to_check_file is not None:
            info_model = pd.read_csv(sample_path + "/" + retr_to_check_file + "/TrainedModel/Information.csv", header=0, sep=",")
            best_retrain = select_over_retrainings(sample_path, selection=selection_criterion, mode="mean", compute_std=compute_std, compute_val=False, rs_val=0)
            info_model["error_train"] = best_retrain["error_train"]
            info_model["train_time"] = best_retrain["train_time"]
            info_model["error_val"] = 0
            info_model["error_test"] = 0
            info_model["L2_norm_test"] = best_retrain["L2_norm_test"]
            info_model["rel_L2_norm"] = best_retrain["rel_L2_norm"]
            if os.path.isfile(sample_path + "/" + retr_to_check_file + "/Images/errors.txt"):
                info_model["l2_glob"] = best_retrain["l2_glob"]
                info_model["l2_glob_rel"] = best_retrain["l2_glob_rel"]
                info_model["l2_om_big"] = best_retrain["l2_om_big"]
                info_model["l2_om_big_rel"] = best_retrain["l2_om_big_rel"]
                info_model["h1_glob"] = best_retrain["h1_glob"]
                info_model["h1_glob_rel"] = best_retrain["h1_glob_rel"]
                try:
                    info_model["l2_p_rel"] = best_retrain["l2_p_rel"]
                except:
                    print("l2_p_rel not found")
                try:
                    info_model["h1_p_rel"] = best_retrain["h1_p_rel"]
                except:
                    print("h1_p_rel not found")
            info_model["setup"] = setup_num
            info_model["retraining"] = best_retrain["retraining"]

            if info_model["batch_size"].values[0] == "full":
                info_model["batch_size"] = best_retrain["Nu_train"] + best_retrain["Nf_train"]
            sensitivity_df = sensitivity_df.append(info_model, ignore_index=True)
        else:
            print(sample_path + "/TrainedModel/Information.csv not found")

    # min_idx = int(np.argmin(criterion))
    # min_idx_test = int(np.argmin(L2_norm))
    # min_test_according_val = L2_norm[min_idx]
    # min_test_according_test = L2_norm[min_idx_test]
    # best_setup = best_retrain_list[min_idx]
    # best = best_retrain_list[min_idx_test]
    # print(min_test_according_val/mean_val*100, best_setup)
    # print(min_test_according_test/mean_val*100, best)

    sensitivity_df = sensitivity_df.sort_values(selection_criterion)
    sensitivity_df = sensitivity_df.rename(columns={'L2_norm_test': 'L2'})
    best_setup = sensitivity_df.iloc[0]
    best_setup.to_csv(base_path + "best.csv", header=0, index=False)
    # print(sensitivity_df)
    print("Best Setup:", best_setup["setup"])
    print(best_setup)

    plt.figure()
    plt.grid(True, which="both", ls=":")
    plt.scatter(sensitivity_df[selection_criterion], sensitivity_df["L2"])
    plt.xlabel(r'$\varepsilon_T$')
    plt.ylabel(r'$\varepsilon_G$')
    # plt.show()
    # quit()
    plt.savefig(base_path + "/et_vs_eg.png", dpi=400)

quit()
total_list = list()
var_list = list()
print(var_list)
print("=======================================================")
print("Regularization Parameter**")
params = sensitivity_df["regularization_parameter"].values
var_list.append("regularization_parameter")
params = list(set(params))
params.sort()
df_kernel_param_list = list()
for value in params:
    index_list_i = sensitivity_df.index[sensitivity_df.regularization_parameter == value]
    new_df = sensitivity_df.loc[index_list_i]
    df_kernel_param_list.append(new_df)
    # print(new_df)
total_list.append(df_kernel_param_list)

print("=======================================================")
print("Kernel Regularizer")
params = sensitivity_df["kernel_regularizer"].values
var_list.append("kernel_regularizer")
params = list(set(params))
params.sort()
df_kernel_reg_list = list()
for value in params:
    index_list_i = sensitivity_df.index[sensitivity_df.kernel_regularizer == value]
    new_df = sensitivity_df.loc[index_list_i]
    df_kernel_reg_list.append(new_df)
    # print(new_df)
total_list.append(df_kernel_reg_list)

print("=======================================================")
print("Number of Neurons")
var_list.append("neurons")
params = sensitivity_df["neurons"].values
params = list(set(params))
params.sort()
df_neurons_list = list()
for value in params:
    index_list_i = sensitivity_df.index[sensitivity_df.neurons == value]
    new_df = sensitivity_df.loc[index_list_i]
    df_neurons_list.append(new_df)
    # print(new_df)
total_list.append(df_neurons_list)

print("=======================================================")
print("Number of Hidden layers")
var_list.append("hidden_layers")
params = sensitivity_df["hidden_layers"].values
params = list(set(params))
params.sort()
df_layers_list = list()
for value in params:
    index_list_i = sensitivity_df.index[sensitivity_df.hidden_layers == value]
    new_df = sensitivity_df.loc[index_list_i]
    df_layers_list.append(new_df)
    # print(new_df)
total_list.append(df_layers_list)

print("=======================================================")
print("Residual regularization")
var_list.append("residual_parameter")
params = sensitivity_df["residual_parameter"].values
params = list(set(params))
params.sort()
df_res_param_list = list()
for value in params:
    index_list_i = sensitivity_df.index[sensitivity_df.residual_parameter == value]
    new_df = sensitivity_df.loc[index_list_i]
    df_res_param_list.append(new_df)
    # print(new_df)
total_list.append(df_res_param_list)

print("=======================================================")
print("Batch Size")
var_list.append("batch_size")
params = sensitivity_df["batch_size"].values
params = list(set(params))
params.sort()
df_bs_list = list()
for value in params:
    index_list_i = sensitivity_df.index[sensitivity_df.batch_size == value]
    new_df = sensitivity_df.loc[index_list_i]
    df_bs_list.append(new_df)
    # print(new_df)
total_list.append(df_bs_list)

var_list_name = [r'$\lambda_{reg}$', r'$q$', r'$d$', r'$K-1$', r'$\lambda$', r'$B$']

if not b:
    out_var_vec = list()
    out_var_vec.append("L2")

    for out_var in out_var_vec:
        for j in range(len(total_list)):
            print("-------------------------------------------------------")
            var = var_list[j]
            var_name = var_list_name[j]
            print(var)
            # name = name_list[j]
            sens_list = total_list[j]
            Nf_dep_fig = plt.figure()
            axes = plt.gca()
            max_val = 0
            plt.grid(True, which="both", ls=":")
            for i in range(len(sens_list)):
                df = sens_list[i]
                print(df)

                value = df[var].values[0]
                label = var_name + r' $=$ ' + str(value)

                sns.distplot(df[out_var], label=label, kde=True, hist=False, norm_hist=False, kde_kws={'shade': True, 'linewidth': 2})
            plt.xlabel(r'$\varepsilon_G$')

            plt.legend(loc=1)
            plt.savefig(base_path + "/Sensitivity_" + var + ".png", dpi=500)

plt.figure()
plt.scatter(sensitivity_df.batch_size, sensitivity_df.L2)
plt.savefig(base_path + "/Sensitivity_batches.png", dpi=500)
plt.show()
quit()
