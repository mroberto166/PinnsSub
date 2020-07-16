from ImportFile import *

pi = math.pi
c = [0, -1, 0]

extrema_values = torch.tensor([[0, 1],
                               [-1, 1]])


def compute_res(network, x_f_train, space_dimensions, solid_object, computing_error=False):
    x_f_train.requires_grad = True
    u = network(x_f_train).reshape(-1, )
    grad_u = torch.autograd.grad(u, x_f_train, grad_outputs=torch.ones(x_f_train.shape[0], ), create_graph=True)[0]
    grad_u_t = grad_u[:, 0]

    grad_u_x = grad_u[:, 1]
    grad_grad_u_x = torch.autograd.grad(grad_u_x, x_f_train, grad_outputs=torch.ones(x_f_train.shape[0]), create_graph=True)[0]
    grad_u_xx = grad_grad_u_x[:, 1]
    residual = grad_u_t.reshape(-1, ) - grad_u_xx.reshape(-1, )

    return residual


def exact(x):
    func = 0
    for n in range(0, len(c)):
        func = func + c[n] * torch.exp(-(n * pi) ** 2 * x[:, 0]) * torch.sin(n * pi * x[:, 1])
    return func.reshape(-1, 1)


def ub0(t):
    # Impose certain type of BC per variable
    # First element is the the index variable
    # Second element is the type of BC for the variable defined by the first element
    type_BC = ["func"]
    out = torch.tensor(()).new_full(size=(t.shape[0], 1), fill_value=0.0)
    return out.reshape(-1, 1), type_BC


def ub1(t):
    type_BC = ["func"]
    out = torch.tensor(()).new_full(size=(t.shape[0], 1), fill_value=0.0)
    return out.reshape(-1, 1), type_BC


list_of_BC = [[ub0, ub1]]


def u0(x):
    func0 = 0
    for n in range(0, len(c)):
        func0 = func0 + c[n] * torch.sin(n * pi * x)
    return func0.reshape(-1, 1)


def convert(vector, extrema_values):
    vector = np.array(vector)
    max_val = np.max(np.array(extrema_values), axis=1)
    min_val = np.min(np.array(extrema_values), axis=1)
    vector = vector * (max_val - min_val) + min_val
    return torch.from_numpy(vector).type(torch.FloatTensor)


def compute_generalization_error(model, extrema, images_path=None):
    model.eval()
    test_inp = convert(torch.rand([100000, extrema.shape[0]]), extrema)
    Exact = (exact(test_inp)).numpy()
    test_out = model(test_inp).detach().numpy()
    assert (Exact.shape[1] == test_out.shape[1])
    L2_test = np.sqrt(np.mean((Exact - test_out) ** 2))
    print("Error Test:", L2_test)

    rel_L2_test = L2_test / np.sqrt(np.mean(Exact ** 2))
    print("Relative Error Test:", rel_L2_test)

    if images_path is not None:
        plt.figure()
        plt.grid(True, which="both", ls=":")
        plt.scatter(Exact, test_out)
        plt.xlabel(r'Exact Values')
        plt.ylabel(r'Predicted Values')
        plt.savefig(images_path + "/Score.png", dpi=400)
    return L2_test, rel_L2_test


def plotting(model, images_path, extrema, solid):
    model.cpu()
    model = model.eval()
    x = torch.reshape(torch.linspace(extrema[1, 0], extrema[1, 1], 100), [100, 1])
    time_steps = [0.0, 0.01, 0.05, 0.1]
    scale_vec = np.linspace(0.65, 1.55, len(time_steps))

    fig = plt.figure()
    plt.grid(True, which="both", ls=":")
    for val, scale in zip(time_steps, scale_vec):
        plot_var = torch.cat([torch.tensor(()).new_full(size=(100, 1), fill_value=val), x], 1)
        plt.plot(x, exact(plot_var), 'b-', linewidth=2, label=r'Exact, $t=$' + str(val) + r'$s$', color=lighten_color('grey', scale), zorder=0)
        plt.scatter(plot_var[:, 1].detach().numpy(), model(plot_var).detach().numpy(), label=r'Predicted, $t=$' + str(val) + r'$s$', marker="o", s=14,
                    color=lighten_color('C0', scale), zorder=10)

    plt.xlabel(r'$x$')
    plt.ylabel(r'u')
    plt.legend()
    plt.savefig(images_path + "/Samples.png", dpi=500)
