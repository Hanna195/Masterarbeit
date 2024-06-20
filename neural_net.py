import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
import os

from polytope import Polytope
from break_points import count_break_points

GRID = 50
BOUNDARY = 10
THRESHOLD_VALUE = 0
MEAN = 0
STD_VAR = 1


class Type(Enum):
    RELU = 1
    THRESHOLD = 2
    SHORTCUT = 3
    LINEAR = 4


class Neural_Net():
    def __init__(self, num_neurons: list[int], activation: Type, evaluation: Type, seed: int) -> None:
        """
        Define and initialize the neural net.

        Args:
            num_neurons (list[int]): number of neurons in each layer
            activation (Type): type of function that is used in hidden layer
            evaluation (Type): type of function that is used in output layer
            seed (int): seed for better reproducibility
        """
        self.layer = len(num_neurons)-1
        self.neurons = num_neurons
        self.activation = activation
        self.evaluation = evaluation
        self.build(seed)

    def build(self, seed: int) -> None:   
        """
        Build the net. Weights are set randomly.

        Args:
            seed (int): Seed for random functions.
        """
        self.seed = seed 
        rnd_gen = np.random.default_rng(seed)
        self.weights = []
        self.biasses = []
        for i in range (self.layer):
            weight_i = rnd_gen.normal(MEAN, STD_VAR, size=(self.neurons[i], self.neurons[i+1]))
            self.weights.append(weight_i)
            self.biasses.append(rnd_gen.random((self.neurons[i+1])))
        if self.activation.name == 'SHORTCUT':
            self.a = rnd_gen.normal(MEAN, STD_VAR, size=(self.neurons[0], self.neurons[self.layer]))
            self.b = rnd_gen.normal(MEAN, STD_VAR, size=(self.neurons[self.layer]))
        print(self.weights)
        print(self.biasses)

    def activation_function(self, x):
        """
        Calculate activation in hidden layer for given value x.

        Args:
            x (float): value of hidden layer before activation.

        Returns:
            float: value of hidden layer after activation.
        """
        if self.activation.name == 'RELU':
            try:
                for j in range(len(x)):
                    x[j] = max(0,x[j]) 
            except TypeError:
                x = max(0, x) 
        elif self.activation.name == 'THRESHOLD' or self.activation.name == 'SHORTCUT':
            for j in range(len(x)):
                if x[j] > THRESHOLD_VALUE:
                    x[j] = 1
                else:
                    x[j] = 0
        elif self.activation.name == 'LINEAR':
            pass
        else:
            print('ERROR! Activation Function not well defined!')
        return x

    def evaluation_function(self, x):
        """
        Calculate evaluation in output layer for given value x.

        Args:
            x (float): value of output layer before evaluation.

        Returns:
            float: value of output layer after evaluation.
        """
        if self.evaluation.name == 'RELU':
            try:
                for j in range(len(x)):
                    x[j] = max(0,x[j]) 
            except TypeError:
                x = max(0, x) 
        elif self.evaluation.name == 'THRESHOLD':
            for j in range(len(x)):
                if x[j] > THRESHOLD_VALUE:
                    x[j] = 1
                else:
                    x[j] = 0
        elif self.evaluation.name == 'LINEAR':
            pass
        else:
            print('ERROR! Evaluation Function not well defined!')
        return x

    def forward_single_input(self, input, stop_layer = None):
        """
        Calculate net output for single input vector.

        Args:
            input (_type_): input vector to net

        Returns:
            _type_: output vector from net
        """
        if stop_layer == None: 
            stop_layer = self.layer
        x_in = input
        for i in range(stop_layer):
            x_out = np.dot(x_in, self.weights[i])
            x_out = x_out + self.biasses[i]
            if i < self.layer-1:
                x_out = self.activation_function(x_out)
            else:
                x_out = self.evaluation_function(x_out)
            x_in = x_out
        if self.activation.name == 'SHORTCUT':
            factor =  np.dot(input, self.a)
            factor = factor + self.b
            x_out = np.dot(factor, x_in)
        return x_out
    
    def forward(self, input: Polytope) -> list[list]:
        """
        Calculate net output for given input polytpe. To get an idea 
          of the whole polytpe, points on a grid are used as single
          input points.

        Args:
            input (Polytpe): polytpe that describes input 

        Returns:
            list[list]: output values from the net
        """
        output_x = []
        output_y = []
        if input.closed:
            x_linspace = np.linspace(input.x_min,input.x_max,GRID)
            y_linspace = np.linspace(input.y_min,input.y_max,GRID)
        else:
            x_linspace = np.linspace(-BOUNDARY,BOUNDARY,GRID)
            y_linspace = np.linspace(-BOUNDARY,BOUNDARY,GRID)

        for x_in in x_linspace:
            for y_in in y_linspace:
                if input.check_point([x_in,y_in]):
                    [x_out, y_out] = self.forward_single_input([x_in,y_in])
                    output_x.append(x_out)
                    output_y.append(y_out)

        return [output_x, output_y]
    
    def forward_bunch_of_points(self, input_x, input_y) -> list[list]:
        """
        Calculate net output for given input points. 

        Args:
            input_x (_type_): x values input points 
            input_y (_type_): y values input points

        Returns:
            list[list]: output values from the net
        """
        assert len(input_x)==len(input_y)
        output_x = []
        output_y = []
        for idx, x_in in enumerate(input_x):
            y_in = input_y[idx]
            [x_out, y_out] = self.forward_single_input([x_in,y_in])
            output_x.append(x_out)
            output_y.append(y_out)
        return [output_x, output_y]
    
    def forward_edge(self, start_point, end_point, steps: int) -> list[list]:
        """
        Calculate net output for points on a line segment between start and end point.

        Args:
            start_point (_type_): start point of line segment
            end_point (_type_): end point of line segment
            steps (int): number of steps on the line segment

        Returns:
            list[list]: output values from the net
        """
        output_x = []
        output_y = []
        for step in range(steps+1):
            x_in = step*1/steps * start_point[0] + (1-(step*1/steps)) * end_point[0]
            y_in = step*1/steps * start_point[1] + (1-(step*1/steps)) * end_point[1]
            [x_out, y_out] = self.forward_single_input([x_in,y_in])
            output_x.append(x_out)
            output_y.append(y_out)
        return [output_x, output_y]

    def plot_all(self, max_break_points_1, max_break_points_2, start=0, end=20):
        """ For a net with input dimension 1 plot the values of all (hidden) neurons
            above the interval start to end.

        Args:
            max_break_points_1 (int): variable to store the maximal number of break points in layer 1
            max_break_points_2 (int): variable to store the maximal number of break points in layer 2
            start (int, optional): Start point of the interval. Defaults to 0.
            end (int, optional): End point of the interva. Defaults to 20.

        Returns:
            _type_: number of maximal break points in the first and second hidden layer
        """
        input = np.linspace(start, end, GRID)
                
        fig, ax = plt.subplots(self.layer, 1, sharex=True)
        fig.tight_layout() 
        for single_layer in range(self.layer):
            num_neurons = self.neurons[single_layer+1]
            # print("Layer", single_layer)
            for hidden in range(num_neurons):
                list = []
                for point in range(len(input)):
                    hidden_point=self.forward_single_input(np.array([input[point]]), stop_layer=single_layer+1)[hidden]
                    list.append(hidden_point)
                
                ax[single_layer].scatter(input, list, label = "Neuron %i"%(hidden+1), s = 10)
                break_points = count_break_points(input, list)
                # print("Neuron",hidden, "Anzahl Knicke:", break_points)
                if single_layer ==1 and break_points > max_break_points_1:
                    max_break_points_1 = break_points
                elif single_layer ==2 and break_points > max_break_points_2:
                    max_break_points_2 = break_points
            ax[single_layer].set_title("Hidden Layer %i"%(single_layer+1))
            ax[single_layer].legend(loc="upper right")
        ax[self.layer-1].set_title("Output Layer")
        # plt.show()
        cwd = os.getcwd()
        path = os.path.join(cwd, 'Images')
        n = len(os.listdir(path))+1
        print("n:", n)
        name = "hidden_"+str(n)
        plt.savefig(os.path.join(path, name))
        plt.close()    

        return max_break_points_1, max_break_points_2
    
    def one_vertex_cutted(self, layer, neuron, polytope) -> bool:
        """ Check wether the tropial hypersurface of the neuron defined
            by layer and neuron cuts through the given polytope.
        """
        for i in range(len(polytope.vertices[0])):
            vertex = [polytope.vertices[0][i], polytope.vertices[1][i]]
            vertex_value = self.forward_single_input(np.array(vertex), stop_layer=layer)[neuron]
            if vertex_value == 0:
                return True
        return False
    
    def neuron_is_active(self, layer, neuron, polytope) -> bool:
        """ Check wether the the neuron defined by layer and neuron is active inside 
            the given polytope, i.e. the output of the neuron is not identical zero.
        """
        test_point_x = sum(polytope.vertices[0])/len(polytope.vertices[0])
        test_point_y = sum(polytope.vertices[1])/len(polytope.vertices[1])
        test_out = self.forward_single_input([test_point_x, test_point_y], layer)
        if test_out[neuron]==0:
            return False
        return True

    def calculate_linear_segments(self, input_polytope: Polytope):
        """ Calculate the subdivision of the given input_polytope into linear regions 
            defined by the neural net

        Returns:
            list: list of subpolytopes/linear regions
        """
        polytope_list = [input_polytope]
        dict = {}
        for layer in range(1, self.layer+1):
            for neuron in range(self.neurons[layer]):
                nr_of_old_polytopes = len(polytope_list)
                for idx in range(nr_of_old_polytopes):
                    polytope = polytope_list[idx]
                    new_weights = np.array([self.weights[layer-1][oldlayer_neuron][neuron] for oldlayer_neuron in range(self.neurons[layer-1])])
                    if (layer-1, 0) in dict:
                        a0 = 0
                        a1 = 0
                        b = self.biasses[layer-1][neuron]
                        for i in range(self.neurons[layer-1]):
                            if self.neuron_is_active(layer-1, i, polytope):
                                a0 += new_weights[i]*dict[layer-1, i][0][0] 
                                a1 += new_weights[i]*dict[layer-1, i][0][1]
                                b += new_weights[i]*dict[layer-1, i][1]
                        a = np.array([a0, a1])
                        b = -b
                    else:
                        a = new_weights
                        b = -self.biasses[layer-1][neuron]
                    dict[layer, neuron] = [a, -b]
                    return_value = self.one_vertex_cutted(layer, neuron, polytope)
                    if return_value:
                        new_polytope1 = polytope.cut(a,b)
                        new_polytope2 = polytope.cut(-a,-b)
                        if new_polytope1.vertices == polytope.vertices or new_polytope2.vertices == polytope.vertices:
                            polytope_list = polytope_list + [polytope]
                        else:
                            polytope_list = polytope_list + [new_polytope1, new_polytope2]
                    else:
                        polytope_list = polytope_list + [polytope]
                polytope_list = polytope_list[nr_of_old_polytopes:]
        
        deletions = []
        for i in range(len(polytope_list)):
            polytope = polytope_list[i]
            if polytope.count_disjoint_vertices() < 3:
                deletions.append(i)
        for i in reversed(deletions):
            del polytope_list[i]
        return polytope_list

                
                


    