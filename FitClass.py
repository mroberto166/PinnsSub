import torch


def fit(Ec, training_set_class, verbose=False, writer=None, folder=None):
    model = Ec.model
    num_epochs = model.num_epochs_opt_LBFGS + model.num_epochs_opt_adam

    freq = 10

    training_coll = training_set_class.data_coll
    training_boundary = training_set_class.data_boundary
    training_initial_internal = training_set_class.data_initial_internal

    model.train()

    def closure():
        optimizer.zero_grad()
        loss_tot, loss_vars, loss_pde = Ec.loss(model, x_u_train_, u_train_, x_b_train_, u_b_train_, x_coll_train_, iteration[0], writer)
        loss_f = torch.log10(loss_tot)
        loss_f.backward()

        train_losses[0] = loss_tot
        train_losses[1] = loss_vars
        train_losses[2] = loss_pde
        if iteration[0] % 500 == 0:
            torch.save(model, folder + "/model.pkl")
        iteration[0] = iteration[0] + 1

        return loss_f

    train_losses = list([torch.tensor(0.), torch.tensor(0.), torch.tensor(0.)])
    iteration = list([0])
    for epoch in range(num_epochs):
        if epoch < model.num_epochs_opt_adam:
            optimizer = model.optimizer_adam
        else:
            print("Setting full batch default option for LBFG. Minibatch and LBFGS together do not work")
            optimizer = model.optimizer_lbfgs
            training_set_class.batch_dim = training_set_class.n_samples
            training_set_class.assemble_dataset()
            training_coll = training_set_class.data_coll
            training_boundary = training_set_class.data_boundary
            training_initial_internal = training_set_class.data_initial_internal

        if verbose and epoch % freq == 0:
            print("################################ ", epoch, " ################################")

        if len(training_boundary) != 0 and len(training_initial_internal) != 0:
            for step, ((x_coll_train_, u_coll_train_), (x_b_train_, u_b_train_), (x_u_train_, u_train_)) in enumerate(zip(training_coll, training_boundary, training_initial_internal)):

                x_coll_train_ = x_coll_train_.to(Ec.device)
                x_b_train_ = x_b_train_.to(Ec.device)
                u_b_train_ = u_b_train_.to(Ec.device)
                x_u_train_ = x_u_train_.to(Ec.device)
                u_train_ = u_train_.to(Ec.device)

                optimizer.step(closure=closure)

                if torch.cuda.is_available():
                    del x_coll_train_
                    del x_b_train_
                    del u_b_train_
                    del x_u_train_
                    del u_train_
                    torch.cuda.empty_cache()

        elif len(training_boundary) == 0 and len(training_initial_internal) != 0:
            for step, ((x_coll_train_, u_coll_train_), (x_u_train_, u_train_)) in enumerate(zip(training_coll, training_initial_internal)):

                x_b_train_ = torch.full((0, x_u_train_.shape[1]), 0)
                u_b_train_ = torch.full((0, x_u_train_.shape[1]), 0)

                x_coll_train_ = x_coll_train_.to(Ec.device)
                x_b_train_ = x_b_train_.to(Ec.device)
                u_b_train_ = u_b_train_.to(Ec.device)
                x_u_train_ = x_u_train_.to(Ec.device)
                u_train_ = u_train_.to(Ec.device)

                optimizer.step(closure=closure)

                if torch.cuda.is_available():
                    del x_coll_train_
                    del x_b_train_
                    del u_b_train_
                    del x_u_train_
                    del u_train_
                    torch.cuda.empty_cache()

        elif len(training_boundary) != 0 and len(training_initial_internal) == 0:
            for step, ((x_coll_train_, u_coll_train_), (x_b_train_, u_b_train_)) in enumerate(zip(training_coll, training_boundary)):

                x_u_train_ = torch.full((0, 1), 0)
                u_train_ = torch.full((0, 1), 0)

                x_coll_train_ = x_coll_train_.to(Ec.device)
                x_b_train_ = x_b_train_.to(Ec.device)
                u_b_train_ = u_b_train_.to(Ec.device)
                x_u_train_ = x_u_train_.to(Ec.device)
                u_train_ = u_train_.to(Ec.device)

                optimizer.step(closure=closure)

                if torch.cuda.is_available():
                    del x_coll_train_
                    del x_b_train_
                    del u_b_train_
                    del x_u_train_
                    del u_train_
                    torch.cuda.empty_cache()

        elif len(training_boundary) == 0 and len(training_initial_internal) == 0:
            for step, (x_coll_train_, u_coll_train_) in enumerate(training_coll):
                if verbose and epoch % freq == 0:
                    print("Batch ", step)

                x_u_train_ = torch.full((0, 1), 0)
                u_train_ = torch.full((0, 1), 0)

                x_b_train_ = torch.full((0, x_u_train_.shape[1]), 0)
                u_b_train_ = torch.full((0, x_u_train_.shape[1]), 0)

                x_coll_train_ = x_coll_train_.to(Ec.device)
                x_b_train_ = x_b_train_.to(Ec.device)
                u_b_train_ = u_b_train_.to(Ec.device)
                x_u_train_ = x_u_train_.to(Ec.device)
                u_train_ = u_train_.to(Ec.device)

                optimizer.step(closure=closure)

                if torch.cuda.is_available():
                    del x_coll_train_
                    del x_b_train_
                    del u_b_train_
                    del x_u_train_
                    del u_train_
                    torch.cuda.empty_cache()
            if epoch % freq == 0:
                print("################################ ", epoch, " ################################")
                print("PDE Residual: ", train_losses[2].detach().cpu().numpy().round(4))

    return train_losses
