import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from itertools import product as cart_prod
import pickle

from sklearn.linear_model import QuantileRegressor

np.random.seed(1)

MODELS = ["triangle", "trapezoid", "fan"]
model = MODELS[2]

def generate(xs, easyness):
    if model == "triangle":
        beta0 = lambda s: s
        beta1 = lambda s: (0.5-s)*easyness
    elif model == "trapezoid":
        beta0 = lambda s: (1-easyness)*s
        beta1 =  lambda s: easyness*s

    if model == "triangle" or model == "trapezoid":
        return np.array([(x, beta0(s) + beta1(s)*x) for (x, s) in cart_prod(xs, np.linspace(0, 1, num=100))])

    else:
        return np.array([(x, st.t.rvs(df = 3, scale = easyness*x)) for (x, y) in cart_prod(xs, np.linspace(0, 1, num=1000))])


def emprisk(theta, xs, ys, tau):
    return sum([(y - theta @ np.c_[1, x][0]) * (tau - (y - theta @ np.c_[1, x][0] < 0)) for (x, y) in zip(xs, ys)])

def omega_thresh(true_beta, xs, ys, tau, alpha):
    indices = np.random.choice(len(ys), len(ys)//2, replace = False)
    train_xs = xs[indices]
    test_xs = np.delete(xs, indices)
    train_ys = ys[indices]
    test_ys = np.delete(ys, indices)

    qr = QuantileRegressor(quantile = tau, alpha = 0).fit(train_xs, train_ys)
    betahat = np.hstack([qr.intercept_, qr.coef_])

    denom = (emprisk(betahat, test_xs, test_ys, tau) - emprisk(true_beta, test_xs, test_ys, tau))

    if denom == 0:
        return np.inf, False

    return np.log(alpha)/denom, denom < 0


def gue(xs, ys, tau, alpha, boot_iters = 100):
    qr = QuantileRegressor(quantile = tau, alpha = 0).fit(xs, ys)
    betahat = np.hstack([qr.intercept_, qr.coef_])

    # Bootstrap to compute omega
    num_omegas = 400
    coverages = np.zeros(num_omegas)
    omegas = np.linspace(1, num_omegas, num=num_omegas)
    for _ in range(boot_iters):
        indices = np.random.choice(len(ys), len(ys))
        boot_xs = xs[indices]
        boot_ys = ys[indices]

        thresh, flip = omega_thresh(betahat, boot_xs, boot_ys, tau, alpha)
        if flip:
            np.add.at(coverages, [x for x in range(num_omegas) if x+1 <= thresh], 1)
        else:
            np.add.at(coverages, [x for x in range(num_omegas) if x+1 >= thresh], 1)

    coverages /= boot_iters
    omega = omegas[np.argmin([abs(alpha - (1-coverage)) for coverage in coverages])]
    omega_coverages = [(np.round(omega, 2), coverage) for (omega, coverage) in zip(omegas, coverages)]
    omega = omega/2
    #print(omega_coverages)
    #print(omega, omega_coverages[np.argmin([abs(alpha - (1-coverage)) for coverage in coverages])])
    #print()

    # Compute GUe value
    indices = np.random.choice(len(ys), len(ys)//2, replace = False)
    train_xs = xs[indices]
    test_xs = np.delete(xs, indices)
    train_ys = ys[indices]
    test_ys = np.delete(ys, indices)

    qr = QuantileRegressor(quantile = tau, alpha = 0).fit(train_xs, train_ys)
    betahat = np.hstack([qr.intercept_, qr.coef_])

    return np.exp(-1*omega * (emprisk(betahat, test_xs, test_ys, tau) - emprisk(np.array([np.quantile(ys, tau), 0]), test_xs, test_ys, tau))), omega

easinesses = np.linspace(0, 1, num = 100)
if model == "triangle" or model == "trapezoid":
    xvals = np.linspace(0, 1, num = 100)
else:
    xvals = [st.chi2.ppf(x, df = 1) for x in np.linspace(0, 1, num = 10000)][:-1]

typeIIrates = []
alpha = 0.1
boot_iters = 100
num_taus = 48
coverages = np.zeros(num_taus)
for easiness in easinesses:
    easiness = 1
    population = generate(xvals, easiness)
    xs, ys = zip(*population)
    plt.scatter(xs, ys)
    plt.show()
    exit()
    coverages = np.zeros(num_taus)
    coverage = 0

    grid = np.linspace(0.02, 0.98, num = num_taus)
    tau_omegas = dict()
    for tau in grid:
        tau_omegas[tau] = []
    for _ in range(boot_iters):
        sample = population[np.random.choice(len(population), 50)]
        xs, ys = zip(*sample)
        xs = np.array(xs).reshape(-1, 1)
        ys = np.array(ys)

        evals = []
        for idx, tau in enumerate(grid):
            q_gue, omega = gue(xs, ys, tau, alpha)
            tau_omegas[tau].append(omega)
            evals.append(q_gue)
            coverages[idx] += q_gue <= 1/alpha
        evals = sorted(evals)[::-1]

        # Checks the GUe-BH coverage
        combined_eval = 0
        for k, e in enumerate(evals):
            combined_eval += (k+1)*e/len(evals)**2
        coverage += combined_eval <= 1/alpha

        # Checks the e-BH FDR
        #coverage += max([k * e/len(evals) for (k, e) in enumerate(evals)]) <= 1/alpha

        num_rejections = sum([k * e/len(evals) >= 1/alpha for (k, e) in enumerate(evals)])


    print(tau_omegas)
    print(easiness, coverage/boot_iters)
    typeIIrates.append(coverage/boot_iters)
    pickle.dump(tau_omegas, open("tau_dict_trapezoid_deltahalf.pkl"))
    break

# Power as function of sample size plot
# Easiness 0.3, family = triangle
#power_ns = [(10.0, 0.97), (20.0, 0.96), (30.0, 0.96), (40.0, 0.94), (50.0, 0.95), (60.0, 0.87), (70.0, 0.88), (80.0, 0.87), (90.0, 0.87), (100.0, 0.84), (110.0, 0.91), (120.0, 0.79), (130.0, 0.88), (140.0, 0.81), (150.0, 0.79), (160.0, 0.77), (170.0, 0.83), (180.0, 0.74), (190.0, 0.67), (200.0, 0.54), (210.0, 0.53), (220.0, 0.46), (230.0, 0.44), (240.0, 0.29), (250.0, 0.33), (260.0, 0.33), (270.0, 0.2), (280.0, 0.15), (290.0, 0.1), (300.0, 0.08), (310.0, 0.06), (320.0, 0.02), (330.0, 0.0), (340.0, 0.01), (350.0, 0.0), (360.0, 0.01), (370.0, 0.0), (380.0, 0.0), (390.0, 0.0), (400.0, 0.0), (410.0, 0.0), (420.0, 0.0), (430.0, 0.0), (440.0, 0.0), (450.0, 0.0), (460.0, 0.0), (470.0, 0.0), (480.0, 0.0), (490.0, 0.0), (500.0, 0.0), (510.0, 0.0), (520.0, 0.0), (530.0, 0.0), (540.0, 0.0), (550.0, 0.0), (560.0, 0.0), (570.0, 0.0), (580.0, 0.0), (590.0, 0.0), (600.0, 0.0), (610.0, 0.0), (620.0, 0.0), (630.0, 0.0), (650.0, 0.0), (660.0, 0.0), (680.0, 0.0), (690.0, 0.0), (710.0, 0.0)]
# Easiness 0.3, family = trapezoid
power_ns = [(10.0, 0.97), (20.0, 0.97), (30.0, 0.96), (40.0, 0.93), (50.0, 0.96), (60.0, 0.88), (70.0, 0.83), (80.0, 0.85), (90.0, 0.86), (100.0, 0.82), (110.0, 0.87), (120.0, 0.82), (130.0, 0.76), (140.0, 0.69), (150.0, 0.52), (160.0, 0.55), (170.0, 0.44), (180.0, 0.4), (190.0, 0.37), (200.0, 0.21), (210.0, 0.13), (220.0, 0.11), (230.0, 0.12), (240.0, 0.13), (250.0, 0.03), (260.0, 0.03), (270.0, 0.01), (280.0, 0.0), (290.0, 0.0), (300.0, 0.0), (310.0, 0.0), (320.0, 0.0), (330.0, 0.0), (340.0, 0.0), (350.0, 0.0), (360.0, 0.0), (370.0, 0.0), (380.0, 0.0), (390.0, 0.0), (400.0, 0.0), (410.0, 0.0), (420.0, 0.0), (430.0, 0.0), (440.0, 0.0), (450.0, 0.0), (460.0, 0.0), (470.0, 0.0), (480.0, 0.0), (490.0, 0.0), (500.0, 0.0)]
xs, ys = zip(*power_ns)
plt.scatter(xs, ys)
plt.xlim((0,400))
plt.xlabel("n")
plt.ylabel("Type II error")
plt.savefig("n_power_" + model + ".png")
plt.clf()

# Learning rates plots
tau_omegas = pickle.load(open("tau_dict_trapezoid_delta0.pkl", "rb"))

xs = [tau for tau in tau_omegas][1:-2]
ys = [np.mean(omegas) for omegas in tau_omegas.values()][1:-2]
errs = [np.var(omegas, ddof=1)**0.5/len(tau_omegas)**0.5 for omegas in tau_omegas.values()][1:-2]
plt.scatter(xs, ys, label = "Δ = 0", marker = ".")
plt.errorbar(xs, ys, yerr=errs, fmt='none')

tau_omegas = pickle.load(open("tau_dict_trapezoid_deltahalf.pkl", "rb"))
ys = [np.mean(omegas) for omegas in tau_omegas.values()][1:-2]
errs = [np.var(omegas, ddof=1)**0.5/len(tau_omegas)**0.5 for omegas in tau_omegas.values()][1:-2]
plt.scatter(xs, ys, label = "Δ = 0.5", marker = "+")
plt.errorbar(xs, ys, yerr=errs, fmt='none', color = 'orange')

tau_omegas = pickle.load(open("tau_dict_trapezoid_delta1.pkl", "rb"))
ys = [np.mean(omegas) for omegas in tau_omegas.values()][1:-2]
errs = [np.var(omegas, ddof=1)**0.5/len(tau_omegas)**0.5 for omegas in tau_omegas.values()][1:-2]
plt.scatter(xs, ys, label = "Δ = 1", marker = "x")
plt.errorbar(xs, ys, yerr=errs, fmt='none', color = 'green')


plt.xlabel("τ")
plt.ylabel("ω")
plt.legend()
plt.ylim((0, 60))
plt.savefig("lr_errs_" + model + ".png")
plt.clf()
exit()

#BH-combined

#Triangle
if model == "triangle":
    TypeIIrates = [(0.0, 0.92), (0.010101010101010102, 0.84), (0.020202020202020204, 0.88), (0.030303030303030304, 0.87), (0.04040404040404041, 0.95), (0.05050505050505051, 0.88), (0.06060606060606061, 0.88), (0.07070707070707072, 0.86), (0.08080808080808081, 0.86), (0.09090909090909091, 0.83), (0.10101010101010102, 0.9),
    (0.11111111111111112, 0.93), (0.12121212121212122, 0.87), (0.13131313131313133, 0.92), (0.14141414141414144, 0.88), (0.15151515151515152, 0.89), (0.16161616161616163, 0.86), (0.17171717171717174, 0.9), (0.18181818181818182, 0.9), (0.19191919191919193, 0.84), (0.20202020202020204, 0.87), (0.21212121212121213, 0.9), (0.22222222222222224, 0.85), (0.23232323232323235, 0.86), (0.24242424242424243, 0.84), (0.25252525252525254, 0.8),
    (0.26262626262626265, 0.81), (0.27272727272727276, 0.83), (0.2828282828282829, 0.86), (0.29292929292929293, 0.89), (0.30303030303030304, 0.84), (0.31313131313131315, 0.81), (0.32323232323232326, 0.8), (0.33333333333333337, 0.75), (0.3434343434343435, 0.88), (0.3535353535353536, 0.82), (0.36363636363636365, 0.72), (0.37373737373737376, 0.84), (0.38383838383838387, 0.66), (0.393939393939394, 0.76), (0.4040404040404041, 0.78), (0.4141414141414142, 0.74),
    (0.42424242424242425, 0.71), (0.43434343434343436, 0.66), (0.4444444444444445, 0.68), (0.4545454545454546, 0.66), (0.4646464646464647, 0.54), (0.4747474747474748, 0.66), (0.48484848484848486, 0.55), (0.494949494949495, 0.55), (0.5050505050505051, 0.55), (0.5151515151515152, 0.46), (0.5252525252525253, 0.52), (0.5353535353535354, 0.51), (0.5454545454545455, 0.48), (0.5555555555555556, 0.44), (0.5656565656565657, 0.45), (0.5757575757575758, 0.48),
    (0.5858585858585859, 0.37), (0.595959595959596, 0.37), (0.6060606060606061, 0.39), (0.6161616161616162, 0.3), (0.6262626262626263, 0.36), (0.6363636363636365, 0.43), (0.6464646464646465, 0.33), (0.6565656565656566, 0.3), (0.6666666666666667, 0.38), (0.6767676767676768, 0.27), (0.686868686868687, 0.26), (0.696969696969697, 0.26),
    (0.7070707070707072, 0.27), (0.7171717171717172, 0.29), (0.7272727272727273, 0.22), (0.7373737373737375, 0.17), (0.7474747474747475, 0.21), (0.7575757575757577, 0.14), (0.7676767676767677, 0.18), (0.7777777777777778, 0.13), (0.787878787878788, 0.19), (0.797979797979798, 0.11), (0.8080808080808082, 0.09), (0.8181818181818182, 0.14),
    (0.8282828282828284, 0.13), (0.8383838383838385, 0.07), (0.8484848484848485, 0.12), (0.8585858585858587, 0.04), (0.8686868686868687, 0.08), (0.8787878787878789, 0.09), (0.888888888888889, 0.01), (0.8989898989898991, 0.03), (0.9090909090909092, 0.02), (0.9191919191919192, 0.0), (0.9292929292929294, 0.03), (0.9393939393939394, 0.03),
    (0.9494949494949496, 0.01), (0.9595959595959597, 0.02), (0.9696969696969697, 0.02), (0.9797979797979799, 0.02), (0.98989898989899, 0.02), (1.0, 0.01)]

#Trapezoid
if model == "trapezoid":
    TypeIIrates = [(0.0, 0.92), (0.010101010101010102, 0.84), (0.020202020202020204, 0.88), (0.030303030303030304, 0.84), (0.04040404040404041, 0.96), (0.05050505050505051, 0.88), (0.06060606060606061, 0.88), (0.07070707070707072, 0.83), (0.08080808080808081, 0.88), (0.09090909090909091, 0.86), (0.10101010101010102, 0.89),
    (0.11111111111111112, 0.9), (0.12121212121212122, 0.83), (0.13131313131313133, 0.83), (0.14141414141414144, 0.79), (0.15151515151515152, 0.77), (0.16161616161616163, 0.82), (0.17171717171717174, 0.87), (0.18181818181818182, 0.83), (0.19191919191919193, 0.83), (0.20202020202020204, 0.77), (0.21212121212121213, 0.75), (0.22222222222222224, 0.78), (0.23232323232323235, 0.73), (0.24242424242424243, 0.69),
    (0.25252525252525254, 0.69), (0.26262626262626265, 0.67), (0.27272727272727276, 0.67), (0.2828282828282829, 0.71), (0.29292929292929293, 0.59), (0.30303030303030304, 0.57), (0.31313131313131315, 0.65), (0.32323232323232326, 0.59), (0.33333333333333337, 0.66), (0.3434343434343435, 0.54), (0.3535353535353536, 0.6), (0.36363636363636365, 0.55), (0.37373737373737376, 0.54), (0.38383838383838387, 0.47),
    (0.393939393939394, 0.38), (0.4040404040404041, 0.49), (0.4141414141414142, 0.31), (0.42424242424242425, 0.22), (0.43434343434343436, 0.37), (0.4444444444444445, 0.4), (0.4545454545454546, 0.36), (0.4646464646464647, 0.25), (0.4747474747474748, 0.26), (0.48484848484848486, 0.24), (0.494949494949495, 0.3), (0.5050505050505051, 0.26), (0.5151515151515152, 0.08), (0.5252525252525253, 0.25),
    (0.5353535353535354, 0.17), (0.5454545454545455, 0.11), (0.5555555555555556, 0.14), (0.5656565656565657, 0.13), (0.5757575757575758, 0.09), (0.5858585858585859, 0.1), (0.595959595959596, 0.1), (0.6060606060606061, 0.11), (0.6161616161616162, 0.08), (0.6262626262626263, 0.07), (0.6363636363636365, 0.05),
    (0.6464646464646465, 0.03), (0.6565656565656566, 0.04), (0.6666666666666667, 0.03), (0.6767676767676768, 0.08), (0.686868686868687, 0.04), (0.696969696969697, 0.01), (0.7070707070707072, 0.04), (0.7171717171717172, 0.0), (0.7272727272727273, 0.04), (0.7373737373737375, 0.02), (0.7474747474747475, 0.03), (0.7575757575757577, 0.02), (0.7676767676767677, 0.01), (0.7777777777777778, 0.03),
    (0.787878787878788, 0.01), (0.797979797979798, 0.0), (0.8080808080808082, 0.01), (0.8181818181818182, 0.0), (0.8282828282828284, 0.0), (0.8383838383838385, 0.0), (0.8484848484848485, 0.0), (0.8585858585858587, 0.0), (0.8686868686868687, 0.01), (0.8787878787878789, 0.0), (0.888888888888889, 0.01), (0.8989898989898991, 0.01),
    (0.9090909090909092, 0.0), (0.9191919191919192, 0.0), (0.9292929292929294, 0.0), (0.9393939393939394, 0.01), (0.9494949494949496, 0.0), (0.9595959595959597, 0.0), (0.9696969696969697, 0.0), (0.9797979797979799, 0.0), (0.98989898989899, 0.0), (1.0, 0.0)]

easinesses = [x for (x, y) in TypeIIrates]
typeIIrates = [y for (x, y) in TypeIIrates]
print([x for x in np.linspace(0, 1, num = 100) if x not in easinesses])
plt.scatter(easinesses, typeIIrates)
plt.xlabel("Δ")
plt.ylabel("Type II error")
plt.ylim(0, 1)
#plt.show()
plt.savefig("powercuve_" + model + ".png")
