from CollectUtils import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
np.random.seed(42)

sensitivity_df = pd.read_csv("ConvergenceAnalysis.csv")
print(sensitivity_df)
print(sensitivity_df.columns)
Nf_list = [1000, 2000, 4000, 8000]
Nu_list = [32, 64, 128, 256]

scale_vec = np.linspace(0.55, 1.65, len(Nf_list))
i = 0
c2 = 0
c1 = np.sqrt(1 + np.exp(1))
fig = plt.figure()
plt.grid(True, which="both", ls=":")
for i, Nf in enumerate(Nf_list):
    print(Nf)
    scale = scale_vec[i]
    index_list_i = sensitivity_df.index[sensitivity_df.Nf_train == Nf]
    new_df = sensitivity_df.loc[index_list_i]
    new_df['N0_train'] = new_df['Nu_train'] / 2
    new_df['Nb0_train'] = new_df['Nu_train'] / 4
    new_df['Nb1_train'] = new_df['Nu_train'] / 4
    # bound = (new_df['error_train'] + 2 * new_df['sigma_res'] / new_df['Nf_train'] ** (1 / 2) + 2 * (new_df['sigma_u'] + new_df['sigma_u_star']) / new_df['Nu_train'] ** (
    #        1 / 2)).values

    bound = np.sqrt(c1 ** 2 * (new_df['loss_u0_train'] ** 2 +
                               new_df['c2_0'] ** 2 * new_df['loss_ub0_train'] ** 2 +
                               new_df['c2_1'] ** 2 * new_df['loss_ub1_train'] ** 2 +
                               new_df['res_loss'] ** 2 +
                               new_df['val_gap_u0'] ** 2 +
                               new_df['c2_0'] ** 2 * new_df['val_gap_ub0'] ** 2 +
                               new_df['c2_1'] ** 2 * new_df['val_gap_ub1'] ** 2 +
                               new_df['val_gap_int'] ** 2 +
                               c1 ** 2 * (new_df['sigma_Ru0'] / new_df['N0_train'] ** 0.5 +
                                          new_df['sigma_Rint'] / new_df['Nf_train'] ** 0.5 +
                                          new_df['c2_0'] ** 2 * np.sqrt(new_df['sigma_Rub0'] / new_df['Nb0_train'] ** 0.5) +
                                          new_df['c2_1'] ** 2 * np.sqrt(new_df['sigma_Rub1'] / new_df['Nb1_train'] ** 0.5))).values)

    # plt.scatter(new_df.Nu.values, new_df.L2_norm_train_sum.values)
    if Nf == 2000:
        plt.scatter(new_df.Nu_train.values, new_df.L2_norm_test.values, label=r'$\EuScript{E}_G$', color=lighten_color('red', scale), zorder=0)
        plt.scatter(new_df.Nu_train.values,
                    np.sqrt(new_df.loss_ub0_train.values ** 2 + new_df.loss_ub1_train.values ** 2 + new_df.loss_u0_train.values ** 2 + new_df.res_loss.values ** 2),
                    label=r'$\EuScript{E}_{T}$', color=lighten_color('gray', scale), zorder=0)
        # plt.scatter(new_df.Nu_train.values, new_df.val_gap_ub0.values + new_df.val_gap_ub1.values + new_df.val_gap_int.values + new_df.val_gap_u0.values, label=r'$\EuScript{E}_{TV}$', color=lighten_color('DarkOrange', scale), zorder=0)
        plt.scatter(new_df.Nu_train.values, bound, label=r'$Bound$', color=lighten_color('blue', scale), zorder=0)
    else:
        plt.scatter(new_df.Nu_train.values, new_df.L2_norm_test.values, color=lighten_color('red', scale), zorder=0)
        plt.scatter(new_df.Nu_train.values,
                    np.sqrt(new_df.loss_ub0_train.values ** 2 + new_df.loss_ub1_train.values ** 2 + new_df.loss_u0_train.values ** 2 + new_df.res_loss.values ** 2),
                    color=lighten_color('gray', scale), zorder=0)
        # plt.scatter(new_df.Nu_train.values, new_df.val_gap_ub0.values + new_df.val_gap_ub1.values + new_df.val_gap_int.values + new_df.res_loss.values,
        #            color=lighten_color('DarkOrange', scale), zorder=0)
        plt.scatter(new_df.Nu_train.values, bound, color=lighten_color('blue', scale), zorder=0)
plt.legend()
plt.xscale("log")
plt.xlabel(r'$N_u$')
plt.yscale("log")
plt.savefig("Conv.png", dpi=400)
plt.show()

quit()
