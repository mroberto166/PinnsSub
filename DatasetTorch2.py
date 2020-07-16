from ImportFile import *

pi = math.pi


class DefineDataset:
    def __init__(self,
                 extrema_values,
                 parameters_values,
                 type_of_coll,
                 n_collocation,
                 n_boundary,
                 n_initial,
                 n_internal,
                 batches,
                 random_seed,
                 output_dimension,
                 n_time_step,
                 space_dimensions=1,
                 time_dimensions=1,
                 obj=None,
                 shuffle=False,
                 type_point_param=None
                 ):
        self.extrema_values = extrema_values
        self.parameters_values = parameters_values
        self.type_of_coll = type_of_coll
        self.space_dimensions = space_dimensions
        self.time_dimensions = time_dimensions
        self.dimensions = time_dimensions + space_dimensions
        self.output_dimension = output_dimension

        self.n_collocation = n_collocation
        self.n_boundary = n_boundary
        self.n_initial = n_initial
        self.n_internal = n_internal
        self.n_time_step = n_time_step

        self.batches = batches
        self.random_seed = random_seed
        if parameters_values is None:
            self.parameter_dimensions = 0
        else:
            self.parameter_dimensions = parameters_values.shape[0]
        self.data = None
        self.data_no_batches = None
        self.obj = obj
        if self.obj is not None:
            self.n_object = obj.n_object_space * obj.n_object_time
        else:
            self.n_object = 0
        self.n_samples = self.n_collocation + 2 * self.n_boundary * self.space_dimensions + self.n_initial * self.time_dimensions + self.n_internal + self.n_object
        self.input_dimensions = self.time_dimensions + self.space_dimensions
        self.BC = None
        self.shuffle = shuffle
        self.type_point_param = type_point_param

        if self.batches == "full":
            self.batches = int(self.n_samples)
        else:
            self.batches = int(self.batches)

        print(n_initial, n_boundary, n_internal)

        # print(self.batches)
        # print(self.n_samples)

    def assemble_dataset(self):

        fraction_coll = int(self.batches * self.n_collocation / self.n_samples)
        fraction_boundary = int(self.batches * 2 * self.n_boundary * self.space_dimensions / self.n_samples)
        fraction_initial = int(self.batches * self.n_initial / self.n_samples)
        fraction_ob = int(self.batches * self.n_object / self.n_samples)

        #############################################

        BC = list()

        if self.extrema_values is not None:

            extrema_0 = self.extrema_values[:, 0]
            extrema_f = self.extrema_values[:, 1]

            x_coll = self.generator_points(self.n_collocation, self.input_dimensions, self.random_seed, self.n_time_step)
            y_coll = torch.from_numpy(np.array([np.nan for _ in range(self.n_collocation)]).reshape(-1, 1)).type(
                torch.FloatTensor)
            for i in range(1, self.output_dimension):
                y_coll = torch.cat([y_coll, torch.from_numpy(
                    np.array([np.nan for _ in range(self.n_collocation)]).reshape(-1, 1)).type(torch.FloatTensor)], 1)

            if self.parameters_values is not None:
                x_param_coll = self.generator_param_samples(self.n_collocation, self.parameter_dimensions, self.random_seed)

            x_coll = x_coll * (extrema_f - extrema_0) + extrema_0
            if self.parameters_values is not None:
                x_param_coll = self.transform_param_data(x_param_coll)
                x_param_coll[x_param_coll < 0] = 0
                x_coll = torch.cat([x_coll, x_param_coll], 1)

            print(x_coll, y_coll)

            if self.n_collocation == 0:
                self.data_coll = DataLoader(torch.utils.data.TensorDataset(x_coll, y_coll), batch_size=1,
                                            shuffle=False)
            else:
                self.data_coll = DataLoader(torch.utils.data.TensorDataset(x_coll, y_coll), batch_size=fraction_coll,
                                            shuffle=self.shuffle)

            x_list_b = list()
            y_list_b = list()

            x_list_b_param = list()
            for i in range(self.time_dimensions, self.time_dimensions + self.space_dimensions):
                BC_01 = list()
                val_0 = np.delete(self.extrema_values, i, 0)[:, 0]
                val_f = np.delete(self.extrema_values, i, 0)[:, 1]
                x_boundary_0 = self.generator_points(self.n_boundary, self.input_dimensions, self.random_seed)

                x_boundary_0[:, i] = torch.tensor(()).new_full(size=(self.n_boundary,), fill_value=0.0)
                x_boundary_0_wo_i = np.delete(x_boundary_0, i, 1)

                if self.parameters_values is not None:
                    x_param_boundary_0 = self.generator_param_samples(self.n_boundary, self.parameter_dimensions, self.random_seed)
                    temporary_inp = self.transform_param_data(x_param_boundary_0)
                    temporary_inp[temporary_inp < 0] = 0
                    [y_boundary_0, type_BC] = Ec.list_of_BC[i - self.time_dimensions][0](
                        torch.cat([x_boundary_0_wo_i * (val_f - val_0) + val_0, temporary_inp], 1))
                    x_list_b_param.append(x_param_boundary_0)
                else:
                    [y_boundary_0, type_BC] = Ec.list_of_BC[i - self.time_dimensions][0](x_boundary_0_wo_i * (val_f - val_0) + val_0)

                BC_01.append(type_BC)
                x_list_b.append(x_boundary_0)
                y_list_b.append(y_boundary_0)
                x_boundary_1 = self.generator_points(self.n_boundary, self.input_dimensions, self.random_seed)
                x_boundary_1[:, i] = torch.tensor(()).new_full(size=(self.n_boundary,), fill_value=1.0)
                x_boundary_1_wo_i = np.delete(x_boundary_1, i, 1)

                if self.parameters_values is not None:
                    x_param_boundary_1 = self.generator_param_samples(self.n_boundary, self.parameter_dimensions,
                                                                      self.random_seed)
                    temporary_inp = self.transform_param_data(x_param_boundary_1)
                    temporary_inp[temporary_inp < 0] = 0
                    [y_boundary_1, type_BC] = Ec.list_of_BC[i - self.time_dimensions][1](
                        torch.cat(
                            [x_boundary_1_wo_i * (val_f - val_0) + val_0, temporary_inp], 1))
                    x_list_b_param.append(x_param_boundary_1)
                else:
                    [y_boundary_1, type_BC] = Ec.list_of_BC[i - self.time_dimensions][1](
                        x_boundary_1_wo_i * (val_f - val_0) + val_0)
                BC_01.append(type_BC)

                BC.append(BC_01)

                x_list_b.append(x_boundary_1)
                y_list_b.append(y_boundary_1)
                # print(x_boundary_0)
                # print(x_boundary_1)

            x_b = torch.cat(x_list_b, 0)
            y_b = torch.cat(y_list_b, 0)

            x_b = x_b * (extrema_f - extrema_0) + extrema_0
            print(x_b, x_b.shape)
            if self.parameters_values is not None:
                x_b_param = torch.cat(x_list_b_param)
                x_b_param = self.transform_param_data(x_b_param)
                x_b_param[x_b_param < 0] = 0
                x_b = torch.cat([x_b, x_b_param], 1)

            if self.n_boundary == 0:
                self.data_boundary = DataLoader(torch.utils.data.TensorDataset(x_b, y_b), batch_size=1,
                                                shuffle=False)
            else:
                self.data_boundary = DataLoader(torch.utils.data.TensorDataset(x_b, y_b), batch_size=fraction_boundary,
                                                shuffle=self.shuffle)

            x_time_0 = self.generator_points(self.n_initial, self.input_dimensions, self.random_seed)
            x_time_0[:, 0] = torch.tensor(()).new_full(size=(self.n_initial,), fill_value=0.0)
            print(x_time_0)
            val_0 = self.extrema_values[1:, 0]
            val_f = self.extrema_values[1:, 1]

            x_time_wo_i = x_time_0[:, 1:]
            if self.parameters_values is not None:
                x_param_time_0 = self.generator_param_samples(self.n_initial, self.parameter_dimensions, self.random_seed)
                # y_time_0 = Ec.u0(torch.cat([x_time_wo_i * (val_f - val_0) + val_0, self.transform_param_data(x_param_time_0)],1))
                x_time_0, y_time_0 = Ec.u0(x_time_wo_i * (val_f - val_0) + val_0)
            else:
                y_time_0 = Ec.u0(x_time_wo_i * (val_f - val_0) + val_0)
            ##################################
            # for Double shear layer
            # x_time, y_time_0 = Ec.u0(x_time_wo_i * (val_f - val_0) + val_0)
            # x_time_0 = torch.cat([x_time_0[:, 0].reshape(-1, 1), x_time], 1)
            ##################################

            # Commented for DSUQ

            x_time_0 = x_time_0 * (extrema_f - extrema_0) + extrema_0
            if self.parameters_values is not None:
                x_param_time_0 = self.transform_param_data(x_param_time_0)
                x_param_time_0[x_param_time_0 < 0] = 0
                x_time_0 = torch.cat([x_time_0, x_param_time_0], 1)

            if self.n_internal != 0:
                x_internal, y_internal = Ec.add_internal_points(self.n_internal)
                self.n_internal = x_internal.shape[0]

                if self.time_dimensions != 0:
                    x_time_internal = torch.cat([x_time_0, x_internal], 0)
                    y_time_internal = torch.cat([y_time_0, y_internal], 0)
                else:
                    x_time_internal = x_internal
                    y_time_internal = y_internal

            else:
                print("not adding")
                x_time_internal = x_time_0
                y_time_internal = y_time_0
                self.n_internal = 0
            # print("############################")
            # print("x_time_internal", x_time_internal, x_time_internal.shape)
            # print("y_time_internal", y_time_internal, y_time_internal.shape)
            # print("############################")
            fraction_internal = int(self.batches * self.n_internal / self.n_samples)
            if fraction_internal == 0 and fraction_initial == 0:
                self.data_initial_internal = DataLoader(torch.utils.data.TensorDataset(x_time_internal, y_time_internal),
                                                        batch_size=1, shuffle=False)
            else:
                self.data_initial_internal = DataLoader(torch.utils.data.TensorDataset(x_time_internal, y_time_internal),
                                                        batch_size=fraction_initial + fraction_internal, shuffle=self.shuffle)

        else:
            x_b, y_b = Ec.add_boundary(2 * Ec.input_dimensions * self.n_boundary)
            x_coll, y_coll = Ec.add_collocations(self.n_collocation)
            x_time_internal, y_time_internal = Ec.add_internal_points(self.n_internal)
            fraction_internal = int(self.batches * self.n_internal / self.n_samples)

            print(x_coll, y_coll.shape)
            print(x_time_internal, y_time_internal.shape)
            print(x_b, y_b.shape)
            # quit()
            if self.n_boundary == 0:
                self.data_boundary = DataLoader(torch.utils.data.TensorDataset(x_b, y_b), batch_size=1,
                                                shuffle=False)
            else:
                self.data_boundary = DataLoader(torch.utils.data.TensorDataset(x_b, y_b), batch_size=fraction_boundary,
                                                shuffle=self.shuffle)

            if self.n_collocation == 0:
                self.data_coll = DataLoader(torch.utils.data.TensorDataset(x_coll, y_coll), batch_size=1,
                                            shuffle=False)
            else:
                self.data_coll = DataLoader(torch.utils.data.TensorDataset(x_coll, y_coll), batch_size=fraction_coll,
                                            shuffle=self.shuffle)

            if fraction_internal == 0 and fraction_initial == 0:
                self.data_initial_internal = DataLoader(torch.utils.data.TensorDataset(x_time_internal, y_time_internal),
                                                        batch_size=1, shuffle=False)
            else:
                self.data_initial_internal = DataLoader(torch.utils.data.TensorDataset(x_time_internal, y_time_internal),
                                                        batch_size=fraction_initial + fraction_internal, shuffle=self.shuffle)

        if self.obj is not None:
            # print("constructing obj")
            x_ob, y_ob, BC_ob = self.obj.construct_object()

            BC.append(BC_ob)

        self.BC = BC

    def generator_param_samples(self, samples, dim, random_seed):

        if self.type_point_param == "uniform":
            torch.random.manual_seed(random_seed)
            return torch.rand([samples, dim]).type(torch.FloatTensor)
        elif self.type_point_param == "normal":
            torch.random.manual_seed(random_seed)
            return torch.randn([samples, dim]).type(torch.FloatTensor)
        elif self.type_point_param == "sobol":
            skip = random_seed
            data = np.full((samples, dim), np.nan)
            for j in range(samples):
                seed = j + skip
                data[j, :], next_seed = sobol_seq.i4_sobol(dim, seed)
            return torch.from_numpy(data).type(torch.FloatTensor)

    def transform_param_data(self, tensor):
        if self.type_point_param == "uniform" or self.type_point_param == "sobol":
            extrema_0 = self.parameters_values[:, 0]
            extrema_f = self.parameters_values[:, 1]
            return tensor * (extrema_f - extrema_0) + extrema_0
        elif self.type_point_param == "normal":
            mean = self.parameters_values[:, 0]
            std = self.parameters_values[:, 1]
            return tensor * std + mean
        else:
            raise ValueError()

    def generator_points(self, samples, dim, random_seed, n_time_step=None):

        if self.type_of_coll == "random":
            torch.random.manual_seed(random_seed)
            return torch.rand([samples, dim]).type(torch.FloatTensor)
        elif self.type_of_coll == "lhs":
            return torch.from_numpy(lhs(dim, samples=samples, criterion='center')).type(torch.FloatTensor)
        elif self.type_of_coll == "grid":
            if dim == 2:
                ratio = (self.extrema_values[0, 1] - self.extrema_values[0, 0]) / (
                        self.extrema_values[1, 1] - self.extrema_values[1, 0])
                sqrt_samples_2 = int(np.sqrt(samples / ratio))
                dir2 = np.linspace(0, 1, sqrt_samples_2)
                sqrt_samples_1 = int(samples / sqrt_samples_2)
                dir1 = np.linspace(0, 1, sqrt_samples_1)
                return torch.from_numpy(
                    np.array([[x_i, y_i] for x_i in dir1 for y_i in dir2]).reshape(dir1.shape[0] * dir2.shape[0],
                                                                                   2)).type(torch.FloatTensor)
            else:
                return torch.linspace(0, 1, samples)
        elif self.type_of_coll == "sobol":
            skip = random_seed
            data = np.full((samples, dim), np.nan)
            for j in range(samples):
                seed = j + skip
                data[j, :], next_seed = sobol_seq.i4_sobol(dim, seed)
            return torch.from_numpy(data).type(torch.FloatTensor)
