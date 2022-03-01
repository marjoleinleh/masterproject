import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc

def init_plotting_preferences():
    # sns.set_context("notebook")
    sns.set_context("paper", font_scale=1.7)
    sns.set_style("white")
    sns.set_palette("colorblind")
    plt.rcParams["figure.figsize"] = 9, 6
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["image.cmap"] = "crest_r"
    rc('font',**{'family':'serif','serif':['Palatino']})
    rc('text', usetex=True)     
    current_cmap = matplotlib.cm.get_cmap()
    current_cmap.set_bad(color="white")
    #plt.rcParams["font.family"] = "cursive"
    # 'TkAgg' crashes on desktop, 'Agg' produces no graphical output
    # matplotlib.use("Qt5Agg")
