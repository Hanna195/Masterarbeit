import numpy as np
import matplotlib.pyplot as plt
from sympy.solvers import solve
from sympy import Symbol
import copy

INPUT_DIMENSION = 2
""" If the polytpe has no vertices: Which section should be viewed/plotted: """
X_MIN = 0
X_MAX = 5
Y_MIN = 0
Y_MAX = 5

class Linear_Functions():
    def __init__(self, A: np.array, b: np.array) -> None:
        """
        Define linear functions / inequalities, defined by matrix and vector as A*x <= b

        Args:
            A (np.array): input matrix
            b (np.array): input vector
        """
        assert A.shape[0] == b.shape[0]
        self.total_number = 0
        self.parameters = []
        for idx in range(np.size(b)):
            self.add_function(A[idx], b[idx])

    def add_function(self, a: np.array, b: float) -> None:
        """
        Add a single function/inequality, defined by a*x <= b. 
        At the moment only possible for a[-1] not equal 0.

        Args:
            a (np.array): input vector
            b (float): boundary
        """
        a_new = []
        if a[-1] != 0:
            for i in range(INPUT_DIMENSION-1):
                a_new.append(-a[i]/a[-1])
            self.parameters.append([a_new, b/a[-1], np.sign(a[-1])])
        else:
            self.parameters.append([a[:-1], b, 0])
        self.total_number += 1

    def call_function(self, index: int, x):
        """
        Calculate f(x), where f is defined through the index of the function.

        Args:
            index (int): index of saved function
            x (_type_): input value

        Returns:
            _type_: output value y = f(x)
        """
        param = self.parameters[index]
        if len(x) != INPUT_DIMENSION-1:
            y = []
            for i in range(len(x)):
                if INPUT_DIMENSION ==2:
                    y_i = self.call_function(index, [x[i]])
                else:
                    y_i = self.call_function(index, x[i])
                y.append(y_i)
            return y
        elif param[-1]==0:
            if type(x[0])==float or type(x[0])==int:
                return - param[1]
            else:
                scalar_product = 0
                for i in range(INPUT_DIMENSION-1):
                    scalar_product += param[0][i]*x[i]
                return scalar_product - param[1]
        elif all(param[0][i]==0 for i in range(len(param[0]))):
            return param[1]*param[2]
        else:
            scalar_product = 0
            for i in range(INPUT_DIMENSION-1):
                scalar_product += param[0][i]*x[i]
            return scalar_product + param[1]
    
    def inequality_parameters(self, index: int):
        """
        return parameters of function/inequality with given index

        Args:
            index (int): index of function/inequality

        Returns:
            _type_: parameters
        """
        param = self.parameters[index]
        return param
    
    def point_in_halfplane(self, point: list, index: int) -> bool:
        """
        Check if given point (x,y) lies in the halfplane defined by inequality with given index 

        Args:
            point (list): input value
            index (int): index of function/inequality

        Returns:
            bool: true if pint lies in halfplane
        """
        if self.parameters[index][2]==0:
            sgn = np.sign(self.parameters[index][0][0])
            x_0 = self.parameters[index][1]
            x = point[0]
            return sgn*x<=sgn*x_0
        else:
            compare_value_y = self.call_function(index, point[:-1])
            if (self.parameters[index][2]==1 and compare_value_y>=point[-1]) or (self.parameters[index][2]==-1 and compare_value_y<=point[-1]):
                return True
        return False


class Polytope():
    def __init__(self, A: np.array, b: np.array) -> None:
        """
        Create a Polytpe defined by A*x <= b

        Args:
            A (np.array): Matrix to define inequalities
            b (np.array): boundaries
        """
        assert A.shape[0] == b.shape[0]
        self.hyperplanes = Linear_Functions(A,b)
        self.edges = [] 
        self.outer_form = [] 
        self.vertices = [[],[]]
        self.closed = False
        self.x_min = X_MIN
        self.x_max = X_MAX
        self.y_min = Y_MIN
        self.y_max = Y_MAX
        self.define_form()
        self.define_vertices()

    def define_form(self):
        """
        Define the Polytpe by adding one inequality after each other

        to the set of hyperplanes that describe the Polytpe.
        On the way: 
            - define the outer_form as edge, vertex, edge, vertex, ...
            - define edges
        """
        x = Symbol('x')
        for i in range(self.hyperplanes.total_number):
            if i>0:
                pass
                # fig, ax = plt.subplots(1,1)
                # ax = self.plot(while_build = True)
                # plt.show()
            self.add_edge(i)


    def add_edge(self, index):
        """ Add one edge to the existing polytope, defined by the hyperplane 
            (linear function) with the given index.
            Calculate the intersection between the new edge and the existing polytope
            and define the outer_form and edges of the polytope in a way that it is 
            described clockwise. 
        """
        x = Symbol('x')
        
        if len(self.edges) == 0:
            self.edges.append([self.hyperplanes.inequality_parameters(index), index])
            self.outer_form.append([self.hyperplanes.inequality_parameters(index), index])

        elif len(self.edges) == 1:
            if self.hyperplanes.parameters[index][2] == 0:
                x_new = self.hyperplanes.parameters[index][0][0]*self.hyperplanes.parameters[index][1]
                y_new = self.hyperplanes.call_function(self.edges[0][1], [x_new])
            elif self.edges[0][0][2]== 0:
                x_new = self.hyperplanes.parameters[self.edges[0][1]][0][0]*self.hyperplanes.parameters[self.edges[0][1]][1]
                y_new = self.hyperplanes.call_function(index, [x_new])
            else:
                try:
                    x_new,  = solve(self.hyperplanes.call_function(index, [x]) - self.hyperplanes.call_function(self.edges[0][1], [x]))
                    y_new = self.hyperplanes.call_function(index, [x_new])
                except ValueError as ve:
                    # propable parallel edges
                    pass
            if self.hyperplanes.point_in_halfplane([x_new+1, self.hyperplanes.call_function(index, [x_new+1])], self.edges[0][1]) != self.hyperplanes.point_in_halfplane([x_new+1, self.hyperplanes.call_function(self.edges[0][1], [x_new+1])], index):
                self.edges =  [[self.hyperplanes.inequality_parameters(index), index], self.edges[0]]
                self.outer_form = [[self.hyperplanes.inequality_parameters(index), index], [x_new, y_new], self.outer_form[0]]
            else:
                self.edges.append([self.hyperplanes.inequality_parameters(index), index])
                self.outer_form.append([x_new, y_new])
                self.outer_form.append([self.hyperplanes.inequality_parameters(index), index])
        
        else:
            new_vertices = []
            intersections = []
            indices = []
            for idx, edge in enumerate(self.edges):
                try:
                    if self.hyperplanes.parameters[index][2] == 0:
                        x_new = self.hyperplanes.parameters[index][0][0]*self.hyperplanes.parameters[index][1]
                        y_new = self.hyperplanes.call_function(edge[1], [x_new])
                    elif edge[0][2] == 0:
                        x_new = edge[0][0][0]*edge[0][1]
                        y_new = self.hyperplanes.call_function(index, [x_new])
                    else:
                        x_new,  = solve(self.hyperplanes.call_function(index, [x]) - self.hyperplanes.call_function(edge[1], [x]))
                        y_new = self.hyperplanes.call_function(index, [x_new])
                    inside = self.check_point([x_new, y_new], idx_list = [edge[1]])
                except ValueError as ve:
                    # propable parallel edges
                    inside = False
                if inside:
                    new_vertices.append([x_new, y_new])
                    intersections.append(edge[1]) 
                    indices.append(idx)
            for i in range(len(indices)-2):
                self.check_for_duplicates(new_vertices, intersections, indices, index)
            if len(indices)==0:
                pass
            elif len(indices)==2:
                point_for_comparison = self.outer_form[2*indices[0]+1]
                if self.hyperplanes.point_in_halfplane(point_for_comparison, index):
                    # point between intersections is inside new hyperplane
                    self.edges=self.edges[indices[0]:indices[1]+1]
                    self.edges.append([self.hyperplanes.inequality_parameters(index), index])
                    
                    self.outer_form=self.outer_form[2*indices[0]:2*indices[1]+1]
                    self.outer_form.append(new_vertices[1])
                    self.outer_form.append([self.hyperplanes.inequality_parameters(index), index])
                    self.outer_form.append(new_vertices[0])
                else:
                    last_part = self.edges[indices[1]:]
                    self.edges=self.edges[:indices[0]+1]
                    self.edges.append([self.hyperplanes.inequality_parameters(index), index])
                    self.edges = self.edges + last_part

                    last_part = self.outer_form[2*indices[1]:]
                    self.outer_form=self.outer_form[:2*indices[0]+1]

                    self.outer_form.append(new_vertices[0])
                    self.outer_form.append([self.hyperplanes.inequality_parameters(index), index])
                    self.outer_form.append(new_vertices[1])
                    
                    self.outer_form = self.outer_form + last_part
                    
            elif len(indices)==1:
                if indices[0]==0:
                    x_new = new_vertices[0][0]-(self.outer_form[1][0]-new_vertices[0][0])
                    check_point = [x_new, self.hyperplanes.call_function(intersections[0], [x_new])]
                    if self.hyperplanes.point_in_halfplane(check_point, index):
                        self.edges = [self.edges[0], [self.hyperplanes.inequality_parameters(index), index]]
                        self.outer_form = [self.outer_form[0], new_vertices[0], [self.hyperplanes.inequality_parameters(index), index]]
                    else:
                        self.edges = [[self.hyperplanes.inequality_parameters(index), index]] + self.edges
                        self.outer_form = [[self.hyperplanes.inequality_parameters(index), index], new_vertices[0]] + self.outer_form
                elif indices[0]==len(self.edges)-1:
                    x_new = new_vertices[0][0]-(self.outer_form[len(self.edges)-1][0]-new_vertices[0][0])
                    check_point = [x_new, self.hyperplanes.call_function(intersections[0], [x_new])]
                    
                    if self.hyperplanes.point_in_halfplane(check_point, index):
                        self.edges = [[self.hyperplanes.inequality_parameters(index), index], self.edges[indices[0]]]
                        self.outer_form = [[self.hyperplanes.inequality_parameters(index), index], new_vertices[0], self.outer_form[2*indices[0]]]
                    else:
                        self.edges = self.edges + [[self.hyperplanes.inequality_parameters(index), index]]
                        self.outer_form = self.outer_form + [new_vertices[0], [self.hyperplanes.inequality_parameters(index), index]]
                else:
                    print("ERROR! Unknown case for index", indices[0],"; Number of Edges:", len(self.edges))

            else:
                print("Error! Indices not length 0, 1 or 2", len(indices))
                print(len(new_vertices), new_vertices)
        if type(self.outer_form[-1][0])!=list:
            self.closed = True

    def check_for_duplicates(self, new_vertices, intersections, indices, hyperplane_index):
        """ Delete duplicate vertices if you calculated more than two intersections 
            between the new edge and the existing polytope (happens if the new edge 
            cuts through a existing vertex)
        """
        for i in range(len(new_vertices)):
            for j in range(i+1,len(new_vertices)):
                if new_vertices[i]==new_vertices[j]:
                    if self.outer_form[2*indices[i]-1] == new_vertices[i]:
                        i_inside = self.hyperplanes.point_in_halfplane(self.outer_form[2*indices[i]+1], hyperplane_index)
                    elif self.outer_form[2*indices[i]+1] == new_vertices[i]:
                        i_inside = self.hyperplanes.point_in_halfplane(self.outer_form[2*indices[i]-1], hyperplane_index)
                    else:
                        print("ERROR! No vertex at hyperplane", indices[i], "is the new vertex!")
                        print(self.outer_form[2*indices[i]-1], self.outer_form[2*indices[i]+1], new_vertices[i])
                        i_inside = True
                    if not i_inside:
                        del new_vertices[i]
                        del intersections[i]
                        del indices[i]
                    else:
                        if self.outer_form[2*indices[j]-1] == new_vertices[j]:
                            j_inside = self.hyperplanes.point_in_halfplane(self.outer_form[2*indices[j]+1], hyperplane_index)
                        elif self.outer_form[2*indices[j]+1] == new_vertices[j]:
                            j_inside = self.hyperplanes.point_in_halfplane(self.outer_form[2*indices[j]-1], hyperplane_index)
                        else:
                            print("ERROR! No vertex at hyperplane", indices[j], "is the new vertex!")
                            print(self.outer_form[2*indices[j]-1], self.outer_form[2*indices[j]+1], new_vertices[j])
                            j_inside = True
                        if j_inside:
                            print("ERROR! Both, i and j, have another vertex inside the new hyperplane")
                            # print(self.hyperplanes.parameters[hyperplane_index])
                            # print(self.outer_form[2*indices[i]-1], self.outer_form[2*indices[i]+1])
                            # print(self.outer_form[2*indices[j]-1], self.outer_form[2*indices[j]+1])
                        else:
                            del new_vertices[j]
                            del intersections[j]
                            del indices[j]
                    return 0

    def check_point(self, point: list, idx_list = []) -> bool:
        """
        Check given point, whether it lays in the Polytpe.

        Args:
            point (list): point to check
            idx_list (list, optional): If given, skip check for inequalities
              with index inside the list to avoid numerical issues. 
              Defaults to [].

        Returns:
            bool: True if point is in Polytpe, False otherwise
        """
        for vert in self.edges:
            if vert[1] not in idx_list:
                if not self.hyperplanes.point_in_halfplane(point, vert[1]):
                    return False
        return True

    def count_disjoint_vertices(self) -> int:
        """ Count the number of disjunct vertices of the polytope
        """
        unique_vertices = []
        for i in range(len(self.vertices[0])):
            vertex = [self.vertices[0][i], self.vertices[1][i]]
            if vertex not in unique_vertices:
                unique_vertices.append(vertex)
        return len(unique_vertices)
    
    def define_vertices(self) -> None:
        """
        Store vertices in self.vertices, 
          also minimum and maximum of x/y values are stored
        """
        x = []
        y = []
        for elem in self.outer_form:
            if type(elem[0])!=list: 
                x.append(float(elem[0]))
                y.append(float(elem[1]))
        self.vertices = [x,y]
        try:
            self.x_min = min(x)
            self.x_max = max(x)
            self.y_min = min(y)
            self.y_max = max(y)
        except ValueError:
            pass

    def cut(self, A, b):
        """ Cut the existing poytope by taking a deepcopy and add the last hyperplane as edge
        """
        new_polytope = copy.deepcopy(self)
        new_polytope.hyperplanes.add_function(A,b)
        new_polytope.add_edge(new_polytope.hyperplanes.total_number-1)
        new_polytope.define_vertices()
        return new_polytope

    def append_vertices(self, x, y):
        """
        Help function for better plots:
          add intersections with plot boundaries and plot corners that
          are inside the Polytpe

        Args:
            x (_type_): x values actual vertices
            y (_type_): y values actual vertices

        Returns:
            _type_: x,y of adapted vertices
        """
        var = Symbol('x')

        x_min = self.x_min-1.0
        x_max = self.x_max+1.0
        y_min = self.y_min-1.0
        y_max = self.y_max+1.0
        
        if not self.closed:
            if len(self.edges)>1:
                # Schnittpunkte erste Ungleichung
                if self.edges[0][0][2]==0:
                    x_set = self.edges[0][0][0][0]*self.edges[0][0][1]
                    if x_set==x_min or x_set==x_max:
                        print("Please check the limits set!")
                    else:
                        x_down = x_set
                        if x_down >= x_min and x_down <= x_max and self.check_point([x_down, y_min], idx_list = [self.edges[0][1]]):
                            indices = [3]
                            x = [x_down]+x
                            y = [y_min]+y
                        # 4. Schnittpunkt mit y_max
                        else:
                            x_up = x_set
                            if x_up >= x_min and x_up <= x_max and self.check_point([x_up, y_max], idx_list = [self.edges[0][1]]):
                                indices = [4]
                                x = [x_up]+x
                                y = [y_max]+y
                            else:
                                print("ERROR! No intersection between first inequality and axes")
                elif self.edges[0][0][0][0]==0:
                    y_set = self.edges[0][0][2]*self.edges[0][0][1]
                    if y_set==y_min or y_set==y_max:
                        print("Please check the limits set!")
                    else: 
                        y_left = y_set
                        if y_left <= y_max and y_left >= y_min and self.check_point([x_min, y_left], idx_list = [self.edges[0][1]]):
                            indices = [1]
                            x = [x_min]+x
                            y = [y_left]+y
                        # 2. Schnittpunkt mit x_max
                        else:
                            y_right = y_set
                            if y_right <= y_max and y_right >= y_min and self.check_point([x_max, y_right], idx_list = [self.edges[0][1]]):
                                indices = [2]
                                x = [x_max]+x
                                y = [y_right]+y
                            else:
                                print("ERROR! No intersection between first inequality and axes")
                else:
                    # 1. Schnittpumkt mit x_min
                    y_left = self.hyperplanes.call_function(self.edges[0][1], [x_min])
                    # print(x_min, y_left)
                    # print(self.check_point([x_min, y_left], idx_list = [self.edges[0][1]]))
                    if y_left <= y_max and y_left >= y_min and self.check_point([x_min, y_left], idx_list = [self.edges[0][1]]):
                        indices = [1]
                        x = [x_min]+x
                        y = [y_left]+y
                    # 2. Schnittpunkt mit x_max
                    else:
                        y_right = self.hyperplanes.call_function(self.edges[0][1], [x_max])
                        if y_right <= y_max and y_right >= y_min and self.check_point([x_max, y_right], idx_list = [self.edges[0][1]]):
                            indices = [2]
                            x = [x_max]+x
                            y = [y_right]+y
                    # 3. Schnittpunkt mit y_min
                        else:
                            x_down = solve(self.hyperplanes.call_function(self.edges[0][1], [var])-y_min, var)[0]
                            if x_down >= x_min and x_down <= x_max and self.check_point([x_down, y_min], idx_list = [self.edges[0][1]]):
                                indices = [3]
                                x = [x_down]+x
                                y = [y_min]+y
                    # 4. Schnittpunkt mit y_max
                            else:
                                x_up = solve(self.hyperplanes.call_function(self.edges[0][1], [var])-y_max, var)[0]
                                if x_up >= x_min and x_up <= x_max and self.check_point([x_up, y_max], idx_list = [self.edges[0][1]]):
                                    indices = [4]
                                    x = [x_up]+x
                                    y = [y_max]+y
                                else:
                                    print("ERROR! No intersection between first inequality and axes")
                
                # Schnittpunkte letzte Ungleichung
                if self.edges[-1][0][2]==0:
                    x_set = self.edges[-1][0][0][0]*self.edges[-1][0][1]
                    if x_set==x_min or x_set==x_max:
                        print("Please check the limits set!")
                    else:
                        x_down = x_set
                        if x_down >= x_min and x_down <= x_max and self.check_point([x_down, y_min], idx_list = [self.edges[-1][1]]):
                            indices.append(3)
                            x.append(x_down)
                            y.append(y_min)
                        # 4. Schnittpunkt mit y_max
                        else:
                            x_up = x_set
                            if x_up >= x_min and x_up <= x_max and self.check_point([x_up, y_max], idx_list = [self.edges[-1][1]]):
                                indices.append(4)
                                x.append(x_up)
                                y.append(y_max)
                            else:
                                print("ERROR! No intersection between first inequality and axes")
                elif self.edges[-1][0][0][0]==0:
                    y_set = self.edges[-1][0][2]*self.edges[-1][0][1]
                    if y_set==y_min or y_set==y_max:
                        print("Please check the limits set!")
                    else: 
                        y_left = y_set
                        if y_left <= y_max and y_left >= y_min and self.check_point([x_min, y_left], idx_list = [self.edges[0][1]]):
                            indices.append(1)
                            x.append(x_min)
                            y.append(y_left)
                        # 2. Schnittpunkt mit x_max
                        else:
                            y_right = y_set
                            if y_right <= y_max and y_right >= y_min and self.check_point([x_max, y_right], idx_list = [self.edges[0][1]]):
                                indices.append(2)
                                x.append(x_max)
                                y.append(y_right)
                            else:
                                print("ERROR! No intersection between first inequality and axes")
                else:
                    # 1. Schnittpumkt mit x_min
                    y_left = self.hyperplanes.call_function(self.edges[-1][1], [x_min])
                    if y_left <= y_max and y_left >= y_min and self.check_point([x_min, y_left], idx_list = [self.edges[-1][1]]):
                        indices.append(1)
                        x.append(x_min)
                        y.append(y_left)
                    # 2. Schnittpunkt mit x_max
                    else:
                        y_right = self.hyperplanes.call_function(self.edges[-1][1], [x_max])
                        if y_right <= y_max and y_right >= y_min and self.check_point([x_max, y_right], idx_list = [self.edges[-1][1]]):
                            indices.append(2)
                            x.append(x_max)
                            y.append(y_right)
                    # 3. Schnittpunkt mit y_min
                        else:
                            x_down = solve(self.hyperplanes.call_function(self.edges[-1][1], [var])-y_min, var)[0]
                            if x_down >= x_min and x_down <= x_max and self.check_point([x_down, y_min], idx_list = [self.edges[-1][1]]):
                                indices.append(3)
                                x.append(x_down)
                                y.append(y_min)
                    # 4. Schnittpunkt mit y_max
                            else:
                                x_up = solve(self.hyperplanes.call_function(self.edges[-1][1], [var])-y_max, var)[0]
                                if x_up >= x_min and x_up <= x_max and self.check_point([x_up, y_max], idx_list = [self.edges[-1][1]]):
                                    indices.append(4)
                                    x.append(x_up)
                                    y.append(y_max)
                                else:
                                    print("ERROR! No intersection between last inequality and axes")
                
                if indices[0] == indices[1]:
                    pass
                elif indices[0] == 1:
                    if indices[1] == 2:
                        x.append(x_max)
                        x.append(x_min)
                        y.append(y_max)
                        y.append(y_max)
                    elif indices[1] == 3:
                        x.append(x_min)
                        y.append(y_min)
                    elif indices[1] == 4:
                        x.append(x_min)
                        y.append(y_max)
                elif indices[0] == 2:
                    if indices[1] == 1:
                        x.append(x_min)
                        x.append(x_max)
                        y.append(y_min)
                        y.append(y_min)
                    elif indices[1] == 3:
                        x.append(x_max)
                        y.append(y_min)
                    elif indices[1] == 4:
                        x.append(x_max)
                        y.append(y_max)
                elif indices[0] == 3:
                    if indices[1] == 1:
                        x.append(x_min)
                        y.append(y_min)
                    elif indices[1] == 2:
                        x.append(x_max)
                        y.append(y_min)
                    elif indices[1] == 4:
                        x.append(x_min)
                        x.append(x_min)
                        y.append(y_max)
                        y.append(y_min)
                elif indices[0] == 4:
                    if indices[1] == 1:
                        x.append(x_min)
                        y.append(y_max)
                    elif indices[1] == 2:
                        x.append(x_max)
                        y.append(y_max)
                    elif indices[1] == 3:
                        x.append(x_min)
                        x.append(x_min)
                        y.append(y_min)
                        y.append(y_max)

            else:               
                if self.edges[0][0][2]== 0:
                    if np.sign(self.edges[0][0][0])==1:
                        x = [x_min, x_min, self.edges[0][0][1], self.edges[0][0][1]]
                    elif np.sign(self.edges[0][0][0])==-1:
                        x = [self.edges[0][0][1], self.edges[0][0][1], x_max, x_max]
                    y = [y_max, y_min, y_min, y_max]
                    return x,y
                
                y_left = self.hyperplanes.call_function(self.edges[0][1], [x_min])
                y_right = self.hyperplanes.call_function(self.edges[0][1], [x_max])
                x_down = solve(self.hyperplanes.call_function(self.edges[0][1], [var])-y_min, var)[0]
                x_up = solve(self.hyperplanes.call_function(self.edges[0][1], [var])-y_max, var)[0]

                indices = []
                                
                if y_left <= y_max and y_left >= y_min:
                    indices.append(1)
                if y_right <= y_max and y_right >= y_min:
                    indices.append(2)
                if x_down >= x_min and x_down <= x_max:
                    indices.append(3)
                if x_up >= x_min and x_up <= x_max:
                    indices.append(4)

                if len(indices) > 0:
                    if indices[0]==1:
                        if indices[1]==2:
                            if self.check_point([x_min, y_min]):   
                                x = [x_min, x_min, x_max, x_max]
                                y = [y_left, y_min, y_min, y_right]
                            else:                             
                                x = [x_max, x_max, x_min, x_min]
                                y = [y_right, y_max, y_max, y_left]
                        elif indices[1]==3:
                            if self.check_point([x_min, y_min]):   
                                x = [x_min, x_min, x_down]
                                y = [y_left, y_min, y_min]
                            else:                              
                                x = [x_down, x_max, x_max, x_min, x_min]
                                y = [y_min, y_min, y_max, y_max, y_left]
                        elif indices[1]==4:
                            if self.check_point([x_min, y_min]):      
                                x = [x_min, x_min, x_max, x_max, x_up]
                                y = [y_left, y_min, y_min, y_max, y_max]
                            else:                                
                                x = [x_up, x_min, x_min]
                                y = [y_max, y_max, y_left]
                    elif indices[0]==2:
                        if indices[1]==3:
                            if self.check_point([x_min, y_min]):   
                                x = [x_max, x_max, x_min, x_min, x_down]
                                y = [y_right, y_max, y_max, y_min, y_min]
                            else:                                 
                                x = [x_down, x_max, x_max]
                                y = [y_min, y_min, y_right]
                        elif indices[1]==4:
                            if self.check_point([x_min, y_min]):      
                                x = [x_up, x_min, x_min, x_max, x_max]
                                y = [y_max, y_max, y_min, y_min, y_right]
                            else:                                
                                x = [x_max, x_max, x_up]
                                y = [y_right, y_max, y_max]
                    else:
                        if self.check_point([x_min, y_min]):         
                            x = [x_up, x_min, x_min, x_down]
                            y = [y_max, y_max, y_min, y_min]
                        else:                                    
                            x = [x_down, x_max, x_max, x_up]
                            y = [y_min, y_min, y_max, y_max]
                else:
                    if self.check_point([x_min, y_min]):   
                        x = [x_min, x_max, x_max, x_min]
                        y = [y_min, y_min, y_max, y_max]
                    else:                                   
                        x = []
                        y = []
        return x,y 

    def plot(self, ax=None, col = None, while_build = False):
        """
        Define a plot of the Polytpe
        Args:
            ax (_type_, optional): axis of (sub)plot. Defaults to None.
            while_build (bool, optional): If set to true, the vertices
              are added manually. Defaults to False.

        Returns:
            _type_: (sub)plot
        """
        if ax is None:
            ax = plt.gca()

        if while_build: 
            self.define_vertices()
        x,y = self.vertices[0], self.vertices[1]
        
        for i in range(len(x)):
            ax.plot(x[i],y[i],c='r', marker = "*",markersize=10)

        if not self.closed:
            x,y = self.append_vertices(x,y)

        if col:
            ax.fill(x, y,col)   
        else:
            ax.fill(x, y,'red',alpha=0.5)   
        
        xr = np.linspace(self.x_min-1.0,self.x_max+1.0,100)

        for edge in self.edges:
            if edge[0][2]==0:
                xr_new = edge[0][0][0]*edge[0][1]*np.ones(100)
                yr = np.linspace(self.y_min-1.0,self.y_max+1.0,100)
                ax.plot(xr_new,yr,'k--')
            else:
                yr = self.hyperplanes.call_function(edge[1], xr)
                ax.plot(xr,yr,'k--')
        if while_build:
            ax.set_xlim(self.x_min-1.0, self.x_max+1.0)
            ax.set_ylim(self.y_min-1.0, self.y_max+1.0)
        else:
            delta_x = min(1, 0.2*(1+np.abs(self.x_min-self.x_max)))
            delta_y = min(1, 0.2*(1+np.abs(self.y_min-self.y_max)))
            ax.set_xlim(self.x_min-delta_x, self.x_max+delta_x)
            ax.set_ylim(self.y_min-delta_y, self.y_max+delta_y)
        return ax
    
    def plot_clear(self, ax=None, while_build = False):
        """
        Define a plot of the Polytpe, but only the edges and vertices are plotted,
        the surface isn't filled.
        Args:
            ax (_type_, optional): axis of (sub)plot. Defaults to None.
            while_build (bool, optional): If set to true, the vertices
              are added manually. Defaults to False.

        Returns:
            _type_: (sub)plot
        """
        if ax is None:
            ax = plt.gca()

        if while_build: 
            self.define_vertices()
        x,y = self.vertices[0], self.vertices[1]
        
        for i in range(len(x)):
            ax.plot(x[i],y[i],c='r', marker = "*",markersize=10)

        if not self.closed:
            x,y = self.append_vertices(x,y)
        
        xr = np.linspace(self.x_min-1.0,self.x_max+1.0,100)

        for edge in self.edges:
            if edge[0][2]==0:
                xr_new = edge[0][0][0]*edge[0][1]*np.ones(100)
                yr = np.linspace(self.y_min-1.0,self.y_max+1.0,100)
                ax.plot(xr_new,yr,'k--')
            else:
                yr = self.hyperplanes.call_function(edge[1], xr)
                ax.plot(xr,yr,'k--')

        delta_x = min(1, 0.2*(np.abs(self.x_min-self.x_max)))
        delta_y = min(1, 0.2*(np.abs(self.y_min-self.y_max)))
        ax.set_xlim(self.x_min-delta_x, self.x_max+delta_x)
        ax.set_ylim(self.y_min-delta_y, self.y_max+delta_y)

        return ax
        
    def easy_plot(self, ax = None, col = 'red', alpha = 1.0):
        """
        Plot only the inner points of the polytope, not the edges and vertices.

        Args:
            ax (_type_, optional): axis of (sub)plot. Defaults to None.
            col (str, optional): Colour. Defaults to 'red'.
            alpha (float, optional): opacity. Defaults to 1.0.

        Returns:
            _type_: _description_
        """
        if ax is None:
            ax = plt.gca()

        x,y = self.vertices[0], self.vertices[1]
        
        for i in range(len(x)):
            ax.plot(x[i],y[i], c='r', marker = "*",markersize=10)

        ax.fill(x, y,col, alpha=alpha)

        return ax


if __name__ == "__main__":
    A = np.array([[2,5],[5,4],[-3,2]])
    b = np.array([3, -2, 0])
    polytpe = Polytope(A,b)
    print("form defined")
    print(polytpe.outer_form)
    fig, ax = plt.subplots(1,1)
    polytpe.plot(ax)
    plt.show()