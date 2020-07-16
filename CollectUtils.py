from ImportFile import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
c = [0, -1, 0]


def exact(x):
    func = 0
    for n in range(0, len(c)):
        func = func + c[n] * torch.exp(-(n * pi) ** 2 * x[:, 0]) * torch.sin(n * pi * x[:, 1])
    return func.reshape(-1, 1)


def der_exact(inputs):
    x = inputs[:, 1]
    t = inputs[:, 0]
    der_x = -pi * torch.exp(-pi ** 2 * t) * torch.cos(pi * x)
    der_t = pi ** 2 * torch.exp(-pi ** 2 * t) * torch.sin(pi * x)
    der = torch.cat([der_t.reshape(-1, 1), der_x.reshape(-1, 1)], 1)
    return der


def get_domain_properites():
    t_ini = 0.0
    t_fin = 1.0
    xb_0 = -1.0
    xb_1 = 1.0

    extrema_values = np.array([[t_ini, t_fin],
                               [xb_0, xb_1]])

    return extrema_values


def select_over_retrainings(folder_path, selection="error_train", mode="min", compute_std=False, compute_val=False, rs_val=0):
    extrema_values = get_domain_properites()
    t_0 = extrema_values[0, 0]
    t_f = extrema_values[0, 1]
    x_0 = extrema_values[1, 0]
    x_f = extrema_values[1, 1]
    retrain_models = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]

    models_list = list()
    for retraining in retrain_models:
        # print("Looking for ", retraining)
        rs = int(folder_path.split("_")[-1])
        retrain_path = folder_path + "/" + retraining
        number_of_ret = retraining.split("_")[-1]

        if os.path.isfile(retrain_path + "/InfoModel.txt"):
            models = pd.read_csv(retrain_path + "/InfoModel.txt", header=0, sep=",")
            models["retraining"] = number_of_ret
            if os.path.isfile(retrain_path + "/Images/errors.txt"):
                errors = pd.read_csv(retrain_path + "/Images/errors.txt", header=0, sep=",")
                models["l2_glob"] = errors["l2_glob"]
                models["l2_glob_rel"] = errors["l2_glob_rel"]
                models["l2_om_big"] = errors["l2_om_big"]
                models["l2_om_big_rel"] = errors["l2_om_big_rel"]
                models["h1_glob"] = errors["h1_glob"]
                models["h1_glob_rel"] = errors["h1_glob_rel"]
                if 'l2_p_rel' in errors.columns:
                    models["l2_p_rel"] = errors["l2_p_rel"]
                if 'h1_p_rel' in errors.columns:
                    models["h1_p_rel"] = errors["h1_p_rel"]
            # models["error_train"] = 10**models["error_train"]
            # models["error_val"] = 0

            if os.path.isfile(retrain_path + "/TrainedModel/model.pkl"):
                trained_model = torch.load(retrain_path + "/TrainedModel/model.pkl")
                trained_model.eval()
                # print(models)
                if compute_std:
                    inp_bound_0 = np.concatenate([np.random.uniform(t_0, t_f, [2000, 1]), np.full((2000, 1), x_0)], axis=1)
                    inp_bound_1 = np.concatenate([np.random.uniform(t_0, t_f, [2000, 1]), np.full((2000, 1), x_f)], axis=1)

                    inp_bound = torch.from_numpy(np.concatenate([inp_bound_0, inp_bound_1], 0)).type(torch.FloatTensor)
                    inp_initial = torch.from_numpy(np.concatenate([np.full((2000, 1), t_0), np.random.uniform(x_0, x_f, [2000, 1])], axis=1)).type(torch.FloatTensor)

                    inp_bound_0 = torch.from_numpy(inp_bound_0).type(torch.FloatTensor)
                    inp_bound_1 = torch.from_numpy(inp_bound_1).type(torch.FloatTensor)

                    inp_bound_0.requires_grad = True
                    inp_bound_1.requires_grad = True
                    # inp = torch.from_numpy(np.concatenate([inp_bound_0, inp_bound_1, inp_initial], axis=0)).type(torch.FloatTensor)
                    # sigma_u = np.std(Ec.exact(inp).numpy())
                    # sigma_u_star = np.std(trained_model(inp).detach().numpy())
                    # models["sigma_u"] = sigma_u
                    # models["sigma_u_star"] = sigma_u_star
                    inp_res = np.random.uniform(0, 1, [2000, 1])
                    inp_res = inp_res * (np.max(extrema_values, axis=1) - np.min(extrema_values, axis=1)) + np.min(extrema_values, axis=1)
                    inp_res = torch.from_numpy(inp_res).type(torch.FloatTensor)
                    residuals = compute_residual(trained_model, inp_res).detach().numpy()
                    sigma_Ru0 = np.std((exact(inp_initial).detach().numpy() - trained_model(inp_initial).detach().numpy()) ** 2)
                    sigma_Rub0 = np.std((exact(inp_bound_0).detach().numpy() - trained_model(inp_bound_0).detach().numpy()) ** 2)
                    sigma_Rub1 = np.std((exact(inp_bound_1).detach().numpy() - trained_model(inp_bound_1).detach().numpy()) ** 2)
                    sigma_Rint = np.std(residuals ** 2)
                    var_u0 = np.var(exact(inp_initial).numpy() ** 2)
                    var_ub = np.var(exact(inp_bound).numpy() ** 2)
                    var_u0_star = np.var(trained_model(inp_initial).detach().numpy() ** 2)
                    var_ub_star = np.var(trained_model(inp_bound).detach().numpy() ** 2)
                    models["sigma_Ru0"] = sigma_Ru0
                    models["sigma_Rub0"] = sigma_Rub0
                    models["sigma_Rub1"] = sigma_Rub1
                    models["sigma_Rint"] = sigma_Rint
                    models["var_u0"] = var_u0
                    models["var_ub"] = var_ub
                    models["var_u0_star"] = var_u0_star
                    models["var_ub_star"] = var_ub_star

                    ub0_pred = trained_model(inp_bound_0)
                    ub0 = exact(inp_bound_0)
                    ub1_pred = trained_model(inp_bound_1)
                    ub1 = exact(inp_bound_1)

                    der_0_pred = torch.autograd.grad(ub0_pred, inp_bound_0, grad_outputs=torch.ones_like(ub0_pred), create_graph=True)[0]
                    der_1_pred = torch.autograd.grad(ub1_pred, inp_bound_1, grad_outputs=torch.ones_like(ub0_pred), create_graph=True)[0]

                    der_0 = der_exact(inp_bound_0)
                    der_1 = der_exact(inp_bound_1)

                    max_0_pred = torch.max(abs(ub0_pred))
                    max_0 = torch.max(abs(ub0))
                    max_der_0_pred = torch.max(torch.max(abs(der_0_pred), 0)[0])
                    max_der_0 = torch.max(torch.max(abs(der_0), 0)[0])

                    max_1_pred = torch.max(abs(ub1_pred))
                    max_1 = torch.max(abs(ub1))
                    max_der_1_pred = torch.max(torch.max(abs(der_1_pred), 0)[0])
                    max_der_1 = torch.max(torch.max(abs(der_1), 0)[0])

                    c2_0 = max_0 + max_der_0 + max_0_pred + max_der_0_pred
                    c2_1 = max_1 + max_der_1 + max_1_pred + max_der_1_pred

                    # print(c2_0,c2_1)

                    models["c2_0"] = c2_0
                    models["c2_1"] = c2_1

                if compute_val:
                    n_coll = int(models["Nf_train"])
                    n_u = int(models["Nu_train"])
                    n_b0 = int(n_u / 4)
                    n_b1 = int(n_u / 4)
                    n_u0 = int(n_u / 2)
                    x_coll_val = generator_points(n_coll, 2, rs_val)
                    x_boundary_0_val = generator_points(n_b0, 2, rs_val)
                    x_boundary_0_val[:, 1] = torch.tensor(()).new_full(size=(n_b0,), fill_value=0.0)
                    x_boundary_1_val = generator_points(n_b1, 2, rs_val)
                    x_boundary_1_val[:, 1] = torch.tensor(()).new_full(size=(n_b1,), fill_value=1.0)
                    x_time_0_val = generator_points(n_u0, 2, rs_val)
                    x_time_0_val[:, 0] = torch.tensor(()).new_full(size=(n_u0,), fill_value=0.0)

                    x_val = torch.cat([x_coll_val, x_boundary_0_val, x_boundary_1_val, x_time_0_val])
                    x_val[:, 1] = x_val[:, 1] * 2 - 1

                    x_coll_vall = x_val[:n_coll, :]
                    x_u_val = x_val[n_coll:, :]

                    res_val = compute_residual(trained_model, x_coll_vall)

                    u_pred_val = trained_model(x_u_val).reshape(-1, )
                    u_val = exact(x_u_val).reshape(-1, )

                    u_pred_b0_val = u_pred_val[n_b0:]
                    u_pred_b1_val = u_pred_val[n_b0:n_b0 + n_b1]
                    u_pred_0_val = u_pred_val[:(n_b0 + n_b1)]

                    u_b0_val = u_val[n_b0:]
                    u_b1_val = u_val[n_b0:n_b0 + n_b1]
                    u_0_val = u_val[:(n_b0 + n_b1)]
                    # loss_u_val = torch.mean((u_pred_val - u_val) ** 2)
                    res_loss_val = round(float(torch.sqrt(torch.mean(res_val ** 2))), 6)
                    loss_ub0_val = round(float(torch.sqrt(torch.mean((u_pred_b0_val - u_b0_val) ** 2))), 6)
                    loss_ub1_val = round(float(torch.sqrt(torch.mean((u_pred_b1_val - u_b1_val) ** 2))), 6)
                    loss_u0_val = round(float(torch.sqrt(torch.mean((u_pred_0_val - u_0_val) ** 2))), 6)

                    # print(res_loss_val, loss_u0_val, loss_ub_val)

                    # loss_reg = regularization(trained_model, 2)
                    # loss_val = float(torch.sqrt(res_loss_val + loss_u_val + 1e-6 * loss_reg))

                    x_coll = generator_points(n_coll, 2, rs)
                    x_boundary_0 = generator_points(n_b0, 2, rs)
                    x_boundary_0[:, 1] = torch.tensor(()).new_full(size=(n_b0,), fill_value=0.0)
                    x_boundary_1 = generator_points(n_b1, 2, rs)
                    x_boundary_1[:, 1] = torch.tensor(()).new_full(size=(n_b1,), fill_value=1.0)
                    x_time_0 = generator_points(n_u0, 2, rs)
                    x_time_0[:, 0] = torch.tensor(()).new_full(size=(n_u0,), fill_value=0.0)

                    x = torch.cat([x_coll, x_boundary_0, x_boundary_1, x_time_0])
                    x[:, 1] = x[:, 1] * 2 - 1

                    x_coll_train = x[:n_coll, :]
                    x_u = x[n_coll:, :]

                    res_train = compute_residual(trained_model, x_coll_train)

                    u_pred = trained_model(x_u).reshape(-1, )
                    u_train = exact(x_u).reshape(-1, )

                    u_pred_b0_train = u_pred[n_b0:]
                    u_pred_b1_train = u_pred[n_b0:n_b0 + n_b1]
                    u_pred_0_train = u_pred[:(n_b0 + n_b1)]

                    u_b0_train = u_train[n_b0:]
                    u_b1_train = u_train[n_b0:n_b0 + n_b1]
                    u_0_train = u_train[:(n_b0 + n_b1)]

                    res_loss = round(float(torch.sqrt(torch.mean(res_train ** 2))), 6)
                    loss_ub0_train = round(float(torch.sqrt(torch.mean((u_pred_b0_train - u_b0_train) ** 2))), 6)
                    loss_ub1_train = round(float(torch.sqrt(torch.mean((u_pred_b1_train - u_b1_train) ** 2))), 6)
                    loss_u0_train = round(float(torch.sqrt(torch.mean((u_pred_0_train - u_0_train) ** 2))), 6)

                    # print(res_loss, loss_u0_train, loss_ub_train)
                    models["res_loss"] = res_loss
                    models["loss_u0_train"] = loss_u0_train
                    models["loss_ub0_train"] = loss_ub0_train
                    models["loss_ub1_train"] = loss_ub1_train

                    models["res_loss_val"] = res_loss_val
                    models["loss_u0_val"] = loss_u0_val
                    models["loss_ub0_val"] = loss_ub0_val
                    models["loss_ub1_val"] = loss_ub1_val

                    # loss_u = round(float(torch.mean((u_pred - u_train) ** 2)))

                    # loss_train = float(torch.sqrt(res_loss + loss_u + 1e-6*loss_reg))

                    # print(res_loss, loss_train, loss_val)

            models_list.append(models)
            # print(models)

        else:
            print("No File Found")

    retraining_prop = pd.concat(models_list, ignore_index=True)
    retraining_prop = retraining_prop.sort_values(selection)
    # print("#############################################")
    # print(retraining_prop)
    # print("#############################################")
    # quit()
    if mode == "min":
        # print("#############################################")
        # print(retraining_prop.iloc[0])
        # print("#############################################")
        return retraining_prop.iloc[0]
    else:
        retraining = retraining_prop["retraining"].iloc[0]
        # print("#############################################")
        # print(retraining_prop.mean())
        # print("#############################################")
        retraining_prop = retraining_prop.mean()
        retraining_prop["retraining"] = retraining
        return retraining_prop


def compute_residual(network, input_values):
    input_values.requires_grad = True
    u = network(input_values).reshape(-1, )
    grad_u = torch.autograd.grad(u, input_values, grad_outputs=torch.ones(input_values.shape[0], ), create_graph=True)[0]
    grad_u_t = grad_u[:, 0]
    grad_u_x = grad_u[:, 1]
    grad_grad_u_x = torch.autograd.grad(grad_u_x, input_values, grad_outputs=torch.ones(input_values.shape[0]), create_graph=True)[0]
    grad_u_xx = grad_grad_u_x[:, 1]

    res = grad_u_t.reshape(-1, ) - grad_u_xx.reshape(-1, )
    return res


def generator_points(samples, dim, random_seed):
    torch.random.manual_seed(random_seed)
    return torch.rand([samples, dim])


def regularization(model, p):
    reg_loss = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            reg_loss = reg_loss + torch.norm(param, p)
    return reg_loss
