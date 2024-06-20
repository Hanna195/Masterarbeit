import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import os

from neural_net import Type, Neural_Net, GRID
from polytope import Polytope
from break_points import count_break_points, get_break_points


max_break_points_1 = 0
max_break_points_2 = 0

# Set the colours for plotting subpolytopes
random.seed(0)
get_colors = lambda n: ["#%06x" % random.randint(0, 0xFFFFFF) for _ in range(n)]
COLORS = get_colors(20)


class Plot():
    """ This class provides some functions with building blocks, so that the other 
        functions can produce (output) plots
    """

    def __init__(self, input = None, net = None, output = None, polytope_list = None, fixed_weights = False) -> None:
        self.input = input
        self.net = net
        self.output = output
        if polytope_list:
            self.subpolytopes = True
            self.polytope_list = polytope_list
        else:
            self.subpolytopes = False
            self.polytope_list = False
        self.fixed_weights = fixed_weights

    def metadata(self, ax=None):
        """ print some metadata at given axis ax and return the axis
        """
        if ax is None:
            ax = plt.gca()
        ax.axis('off')
        textstr = 'Architecture: \n'+str(self.net.neurons)+'\n Activation:\n'+self.net.activation.name +'\n Evaluation:\n'+self.net.evaluation.name+'\n Seed: \n'+ str(self.net.seed)
        if self.subpolytopes:
            textstr = textstr + '\n Nr. of Subpolytopes \n' + str(len(self.polytope_list))
        if self.fixed_weights:
            textstr = textstr + '\n Part. fixed weights'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        text = ax.text(0.5, 0.5, textstr, ha='center', va='center', bbox=props)
        text.set_path_effects([path_effects.Normal()])
        return ax

    def output_scatter(self, ax=None, col = 'b', polytope = None):
        
        if ax is None:
            ax = plt.gca()  
        if polytope:
            out = self.net.forward(polytope)
        else:
            out = self.output
        if ax is None:
            ax = plt.gca()    
        ax.scatter(out[0], out[1], c= col, label = 'General output net')
        return ax
    
    def output_vertices(self, ax=None, polytope = None):
        if ax is None:
            ax = plt.gca()  
        if polytope == None:
            polytope = self.input
        [x_vertices, y_vertices] = self.net.forward_bunch_of_points(polytope.vertices[0], polytope.vertices[1])
        ax.scatter(x_vertices, y_vertices, c='r', marker = "*", s=100, label = 'Transformed (original) vertices')
        return ax
    
    def output_edges(self, ax=None, polytope = None, col = 'g'):
        if ax is None:
            ax = plt.gca()  
        if polytope == None:
            polytope = self.input
        x_edges = []
        y_edges = []
        for idx in range(len(polytope.vertices[0])):
            [x_edge, y_edge] = self.net.forward_edge([polytope.vertices[0][idx-1], polytope.vertices[1][idx-1]],
                                                [polytope.vertices[0][idx], polytope.vertices[1][idx]], 
                                                100)
            x_edges = x_edges + x_edge
            y_edges = y_edges + y_edge
        ax.scatter(x_edges, y_edges, c = col, label = 'Transformed edges')
        return ax

    def plot_output(self, ax=None, col = None):
        if ax is None:
            ax = plt.gca()
        if col:
            self.output_scatter(ax, col = col)
        else:
            self.output_scatter(ax)
        self.output_vertices(ax)
        self.output_edges(ax)
        return ax

    def extended_plot(self):
        """ Generate a plot with the input polytope, net output - divided into linear regions - and some net metadata.
        """
        fig, ax = plt.subplots(2,2, gridspec_kw={'width_ratios': [3, 1]})
        colours = COLORS 

        # 1. Polytope
        self.input.plot_clear(ax[0][0])
        for i, polytope in enumerate(self.polytope_list):
            polytope.easy_plot(ax[0][0], colours[i])

        # 2. Net output
        for i, poly in enumerate(self.polytope_list):
            self.output_scatter(ax[1][0], col = colours[i], polytope = poly)
        self.output_vertices(ax[1][0])
        self.output_edges(ax[1][0])
        
        # 3. Metadata
        gs = ax[0,1].get_gridspec()
        for axis in ax[0:, 1]:
            axis.remove()
        axbig = fig.add_subplot(gs[0:, 1])
        self.metadata(axbig)
        
        # 4. Show/Store
        # plt.show()
        cwd = os.getcwd()
        path = os.path.join(cwd, 'Images')
        n = len(os.listdir(path))
        print("n:", n)
        plt.savefig(os.path.join(path, str(n)))
        plt.close()
        
    def plot_subpolytopes(self):
        """ Generate a plot with both, the input polytope and the net output divided 
            into linear regions (subpolytopes), and some net metadata.
        """
        fig, ax = plt.subplots(1,2, gridspec_kw={'width_ratios': [3, 1]})
        colours = COLORS 

        # 1. Polytope
        self.input.plot_clear(ax[0])
        for i, polytope in enumerate(self.polytope_list):
            polytope.easy_plot(ax[0], colours[i], alpha = 0.7)

        # 3. Metadata
        self.metadata(ax[1])
        
        # 4. Show/Store
        #plt.show()
        cwd = os.getcwd()
        path = os.path.join(cwd, 'Images')
        n = len(os.listdir(path))
        print("n:", n)
        plt.savefig(os.path.join(path, str(n)))
        plt.close()

    def plot_convex(self):
        """ Generate a plot with the input polytope both, as a whole polytope and divided
            into linear regions (subpolytopes), net output and some net metadata.
        """
        fig, ax = plt.subplots(2,2)
        colours = get_colors(len(self.polytope_list))
        
        # 1. Polytope
        self.input.plot_clear(ax[0][0])
        for i, polytope in enumerate(self.polytope_list):
            polytope.easy_plot(ax[0][0], colours[i])

        # 2A. Subpolytopes
        for i, polytope in enumerate(self.polytope_list):
            [x_vertices, y_vertices] = self.net.forward_bunch_of_points(polytope.vertices[0], polytope.vertices[1])
            # self.output_scatter(ax[1][1], col = colours[i], polytope = polytope)
            ax[1][0].scatter(x_vertices, y_vertices, marker = "*",c = colours[i], s=100, label = 'Transformed vertices')
            # ax[1][1].scatter(x_vertices, y_vertices, marker = "*",c = colours[i], s=100, label = 'Transformed vertices')
            ax[1][0].fill(x_vertices, y_vertices, c = colours[i])

        # 2B. Net output
        self.plot_output(ax[1][1])
        for i, poly in enumerate(self.polytope_list):
            self.output_scatter(ax[1][1], col = colours[i], polytope = poly)
        self.output_edges(ax[1][1])
        self.output_vertices(ax[1][1])
        
        # 3. Metadata
        self.metadata(ax[0][1])

        # 4. Show/Store
        # plt.show()
        cwd = os.getcwd()
        path = os.path.join(cwd, 'Images')
        n = len(os.listdir(path))
        print("n:", n)
        plt.savefig(os.path.join(path, str(n)))
        plt.close()

    def plot_2D(self, col=None):
        """
        Generate a plot that contains three subplots:
        1. A plot of the input polytpe
        2. The output of the net, given the input polytpe
        3. Some metadata information. Stored and printed for better
        reproducibility and explainability.
        """
        fig, ax = plt.subplots(2,2, gridspec_kw={'width_ratios': [3, 1]})
        # 1. Polytpe
        self.input.plot(ax[0][0], col)

        # 2. Net output
        self.plot_output(ax[1][0], col)

        # 3. Metadata
        gs = ax[0,1].get_gridspec()
        for axis in ax[0:, 1]:
            axis.remove()
        axbig = fig.add_subplot(gs[0:, 1])

        self.metadata(axbig)
        
        # 4. Show/Store
        # plt.show()
        cwd = os.getcwd()
        path = os.path.join(cwd, 'Images')
        n = len(os.listdir(path))
        print("n:", n)
        plt.savefig(os.path.join(path, str(n)))
        plt.close()

    def plot_line_segment(self, x_list, y_list):
        """ Plot the two given lists as scatterplot and add some metadata.

        Args:
            x_list (list(floats)): x coordinates of the points
            y_list (list(floats)): y coordinates of the points
        """
        cm = plt.get_cmap("RdYlGn")
        col = np.arange(GRID+1)
        #col = [cm(float(i)/(GRID+1)) for i in range(GRID+1)]

        # Scatterplot
        fig, ax = plt.subplots(1,2, gridspec_kw={'width_ratios': [3, 1]})
        ax[0].scatter(x_list, y_list, s=10, c=col, marker='o')  
        self.metadata(ax[1])

        #plt.show()
        cwd = os.getcwd()
        path = os.path.join(cwd, 'Images')
        n = len(os.listdir(path))
        print("n:", n)
        plt.savefig(os.path.join(path, str(n)))
        plt.close()

    def plot_gradient(self):
        """ Plot the net output divided into subpolytopes and add two arrows per 
            subpolytope to display the gradients in each subpolytope.
        """
        self.input.plot_clear()
        it0 =True
        for polytope in self.polytope_list:
            mid_x = sum(polytope.vertices[0])/len(polytope.vertices[0])
            mid_y = sum(polytope.vertices[1])/len(polytope.vertices[1])

            next_x = min((x for x in polytope.vertices[0] if x >= mid_x))
            next_y = min((y for y in polytope.vertices[1] if y >= mid_y))

            delta_x = (next_x-mid_x)/2
            delta_y = (next_y-mid_y)/2
            
            ref_value = self.net.forward_single_input([mid_x, mid_y])
            diff_x = (self.net.forward_single_input([mid_x+delta_x, mid_y])-ref_value)/delta_x
            diff_y = (self.net.forward_single_input([mid_x, mid_y+delta_y])-ref_value)/delta_x

            grad_y1 = [diff_x[0], diff_y[0]]
            grad_y2 = [diff_x[1], diff_y[1]]

            plt.fill(polytope.vertices[0],polytope.vertices[1], alpha=0.5)
            plt.scatter(mid_x, mid_y, c = 'r', marker='x')
            plt.arrow(x=mid_x, y=mid_y, dx=grad_y1[0], dy=grad_y1[1], width=.007, facecolor='green', edgecolor='none', label = 'dy1/dx') 
            plt.arrow(x=mid_x, y=mid_y, dx=grad_y2[0], dy=grad_y2[1], width=.007, facecolor='purple', edgecolor='none', label = 'dy2/dx') 
            if it0:
                plt.legend(loc='upper right')
            it0 = False
            
        plt.show()


def line_segment_as_input(start, end, architecture, number = 1, seed = None):
    """
        Use the interval from start to end as input for a Neural Net with
        given architecture, count the break points and plot the output of the net.

    Args:
        start (float): start value of the input interval
        end (float): end value of the input interval
        architecture (list(int)): net architecture, given as number of neurons per layer
        number (int, optional): number of nets that should be built. Defaults to 1.
        seed (int, optional): seed for net creation if wanted. Defaults to None.
    """
    list_of_break_points = []
    for i in range(number):
        seed = seed if seed else random.randint(0,100)
        print()
        print('SEED:', seed)
        net = Neural_Net(num_neurons = architecture, activation = Type.RELU, evaluation = Type.LINEAR, seed = seed)
        x_list = []
        y_list = []
        for i in range(GRID+1):
            value = (1-i*1/GRID) * start + (i*1/GRID) * end
            [x_out, y_out] = net.forward_single_input(np.array([value]))
            x_list.append(x_out)
            y_list.append(y_out)

        break_points = count_break_points(x_list, y_list)
        list_of_break_points.append(break_points)
        print("Number of break points:", break_points)
        plot = Plot(net = net)
        plot.plot_line_segment(x_list, y_list)
    print("-------------------")
    print("max number of  break points:", max(list_of_break_points), "at position/seed", np.argmax(np.array(list_of_break_points)))

def build_net_and_polytope(input: Polytope, architecture, number=1):
    """
        Create nets with the given architecture and use the given input.
        The output of all nets is plotted, if the output dimension of the net is 2.

    Args:
        input (Polytope): input polytope
        architecture (list(int)): net architecture, given as number of neurons per layer
        number (int, optional): number of nets that should be built. Defaults to 1.
    """
    for i in range(number):
        seed = random.randint(0,100)
        print()
        print('SEED:', seed)
        net = Neural_Net(num_neurons = architecture, activation = Type.RELU, evaluation = Type.LINEAR, seed = seed)
        output = net.forward(input)
        if architecture[-1] == 2:
            plot = Plot(input, net, output)
            plot.plot_2D()
        

if __name__=="__main__":
    # #sys.stdout = open('output_knicke.txt','wt')

    architecture = [1, 2, 2, 2]
    
    A = np.array([[-1, 0], [0, -1], [1, 0], [0, 1]])
    b = np.array([0, 0, 3, 3])

    input = Polytope(A,b)

    line_segment_as_input(0, 20, architecture)
        
        