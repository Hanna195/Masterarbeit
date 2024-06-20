import numpy as np

def count_break_points(x, y) -> int:
    """ Count the break points between the given x and y coordinates

    Args:
        x (list(float)): interpreted as x coordinates of a list of points
        y (list(float)): interpreted as y coordinates of a list of points

    Returns:
        int: number of breakpoints
    """
    assert len(x)==len(y)
    break_points = 0
    last_break_points = 0
    for i in range(1, len(x)):
        diff_x = x[i]-x[i-1]
        diff_y = y[i]-y[i-1]
        if diff_y != 0:
            new_ratio = diff_x/diff_y
        else:
            new_ratio = 0
        if i == 1:
            old_ratio = new_ratio
        elif np.abs(old_ratio - new_ratio)>1e-5:
            if last_break_points != i-1:
                last_break_points = i
                break_points += 1
            old_ratio = new_ratio
    return break_points


def get_break_points(x, y):
    """ Calculate the break points between the given x and y coordinates

    Args:
        x (list(float)): interpreted as x coordinates of a list of points
        y (list(float)): interpreted as y coordinates of a list of points

    Returns:
        _type_: coordinates of the break points
    """
    assert len(x)==len(y)
    break_points_x = []
    break_points_y = []
    last_break_points = 0
    for i in range(1, len(x)):
        diff_x = x[i]-x[i-1]
        diff_y = y[i]-y[i-1]
        if diff_y != 0:
            new_ratio = diff_x/diff_y
        else:
            new_ratio = 0
        if i == 1:
            old_ratio = new_ratio
        elif np.abs(old_ratio - new_ratio)>1e-5:
            # print(x[i], y[i])
            if last_break_points != i-1:
                last_break_points = i
                break_points_x.append(x[i])
                break_points_y.append(y[i])
            old_ratio = new_ratio
    return break_points_x, break_points_y
