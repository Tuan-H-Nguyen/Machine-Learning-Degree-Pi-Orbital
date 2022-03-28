import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FormatStrFormatter
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple


#plt font initialize
annotate = {'fontname':'Times New Roman','weight':'bold','size':13}
tick = {'fontname':'Times New Roman','size':13}
font = FontProperties()
font.set_weight('bold')
font_legend = font_manager.FontProperties(family = 'Times New Roman',size = 12)
  
def plot_histogram(
    dataset,
    save_path=None,
    label=None,
    label_loc=None,
    x_labels=True
    ):
    fig,ax = plt.subplots(nrows=1,ncols=1)
    ax.hist(
        dataset.loc[:]['Egap'],
        bins=[1.5,2,2.5,3,3.5,4,4.5,5]
        )
    ax.set_ylabel('Number of samples (samples)',**annotate)
    #
    ax.yaxis.set_major_locator(plt.MultipleLocator(10))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(5))
    #
    labely = ax.get_yticks().tolist()
    ax.yaxis.set_ticklabels(labely,**tick)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.f'))  
    #
    if x_labels:
        ax.set_xlabel('Band gap (eV)',**annotate)
        labelx = ax.get_xticks().tolist()
        ax.xaxis.set_ticklabels(labelx,**tick)
    else:
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.xaxis.set_minor_locator(plt.NullLocator())

    if label != None and label_loc != None:
        x,y = label_loc
        ax.text(x,y,label,**annotate)
    if save_path != None:
        fig.savefig(save_path,dpi=600,bbox_inches='tight')

class scatter_plot:
    def __init__(self,double_ax = False,figsize = None):
        if figsize :
            assert isinstance(figsize,tuple)
            self.fig, self.ax = plt.subplots(
                nrows=1, ncols=1,
                figsize=figsize)
        else: 
            self.fig, self.ax = plt.subplots(nrows=1, ncols=1)
        if double_ax:
            self.second_ax = self.ax.twinx()
        self.lines = []
        self.scatters = []

    def add_plot(
        self,
        x,y,
        xlabel=None,
        ylabel=None,
        second_ax = False,
        scatter = True,
        plot_line=None,
        weight=None,
        i=None,
        xticks_format=2,
        yticks_format=2,
        x_minor_tick=None,
        x_major_tick=None,
        y_minor_tick=None,
        y_major_tick=None,
        xlim=None, ylim=None,
        line_color = None,
        line_type = None,
        linewidth = 1.5,
        scatter_color=None,
        scatter_marker = None,
        scatter_size = 15,
        label =None,
        line_label = None,
        equal_aspect = False,
        tick_color = None
        ):
        if second_ax == False:
            ax = self.ax
        else:
            ax = self.second_ax
        if scatter:
            scat = ax.scatter(x,y,c = scatter_color, label = label,s=scatter_size, marker = scatter_marker)
            self.scatters.append(scat)
        if plot_line:
            if weight == None:
                line, = ax.plot(x,y,linewidth=linewidth, c=line_color, linestyle=line_type,label=line_label)
                self.lines.append(line)
            else:
                assert isinstance(weight,tuple)
                wb,w = weight
                if i == None:
                    i = np.linspace(min(x),max(x),1000)
                else:
                    assert isinstance(i,tuple)
                    i = np.linspace(i[0],i[1],100)
                line, = ax.plot(i,wb+w*i,linewidth=linewidth, c=line_color, linestyle=line_type, label = line_label)
                self.lines.append(line)
        if equal_aspect:
            self.fig.gca().set_aspect('equal',adjustable='box')
        #
        ax.set_xlabel(xlabel,**annotate)
        if xlim:
            x,y = xlim
            ax.set_xlim(x,y)
        if type(x_major_tick) is float or type(x_major_tick) is int:
            ax.xaxis.set_major_locator(plt.MultipleLocator(x_major_tick))
        elif x_major_tick == 'null':
            ax.xaxis.set_major_locator(plt.NullLocator())
        if x_minor_tick:
            ax.xaxis.set_minor_locator(plt.MultipleLocator(x_minor_tick))
        try:
            labelx = ax.get_xticks().tolist()
            ax.xaxis.set_ticklabels(labelx,**tick)
            xticks_format = '%.f' if xticks_format==0 else '%.'+str(xticks_format)+'f'
            ax.xaxis.set_major_formatter(FormatStrFormatter(xticks_format))
        except AttributeError:
            pass
            #
        ax.set_ylabel(ylabel,**annotate)
        if ylim:
            x,y = ylim
            ax.set_ylim(x,y)
        if type(y_major_tick) is float or type(y_major_tick) is int:
            ax.yaxis.set_major_locator(plt.MultipleLocator(y_major_tick))
        elif y_major_tick == 'null':
            ax.yaxis.set_major_locator(plt.NullLocator())
        if y_minor_tick:
            ax.yaxis.set_minor_locator(plt.MultipleLocator(y_minor_tick))
        labely = ax.get_yticks().tolist()
        ax.yaxis.set_ticklabels(labely,**tick)
        yticks_format = '%.f' if yticks_format==0 else '%.'+str(yticks_format)+'f'
        ax.yaxis.set_major_formatter(FormatStrFormatter(yticks_format))
        if tick_color:
            ax.tick_params(axis='y',labelcolor=tick_color)
        
    def add_text(self,x,y,text):
        self.ax.text(x,y,text,**annotate) 

    def add_legend(self,loc = None,ncols=None):
        if loc == "None":
            self.ax.legend(prop = font_legend)
        elif loc == "above outside":
            self.ax.legend(
                prop = font_legend,
                loc="lower left",
                bbox_to_anchor=(0,1.02,1,0.2),
                mode="expand", borderaxespad=0,
                ncol = ncols
                )
        elif loc == 'left outside':
            self.ax.legend(
                loc = "center left",
                bbox_to_anchor=(1.04,0.5), borderaxespad=0)


    def save_fig(self,save_path,dpi=600):
        #self.ax.legend()
        self.fig.savefig(save_path,dpi=dpi,bbox_inches="tight")

    def clear(self):
        self.fig.clf()
        del self.fig
