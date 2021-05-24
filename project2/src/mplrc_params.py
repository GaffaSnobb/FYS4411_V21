import matplotlib as mpl
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=[
    "tab:blue", "tab:green", "tab:red", "tab:purple", "tab:orange",
    "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"
])
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['legend.fontsize'] = 15
mpl.rcParams['axes.labelsize'] = 15
mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['ytick.labelsize'] = 15
# mpl.rcParams['text.usetex'] = True