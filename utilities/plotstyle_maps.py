from matplotlib import colors

def get_color_map_selection():
    """Specify colors to use for visualizing mnist"""
    cmap = {}
    cmap['color_black'] = '#5A5A5A'
    cmap['color_red'] = '#CD3333'
    cmap['color_yel'] = '#E3CF57'
    cmap['color_cyan'] = '#42c9c9ff'
    cmap['color_lightblue'] = '#3380f2'
    cmap['color_darkgray'] = '#838B8B'
    cmap['color_green'] = '#40826d' #'#aaffdd' # #003700
    cmap['color_lime'] = '#ddffbb'
    cmap['color_pink'] = '#FFAEB9'
    cmap['color_lightgreen'] = '#c0ff0c'
    cmap['color_orange'] = '#f5a565'
    cmap['color_darkgreen'] = '#40826d'
    cmap['color_darkblue'] = '#00008B'
    cmap['white'] = '#FFFFFF'
    cmap['transparent'] = '#FFFFFF00'
    return cmap

def get_color_map_models():
    colors = get_color_map_selection()
    cmap = {}
    cmap['Reference perturbations'] = colors['color_cyan']
    cmap['reference'] = colors['color_black']
    cmap['falkon'] = colors['color_darkblue']
    cmap['streamrak'] = colors['color_red']
    cmap['euc-knn1'] = colors['color_darkgray']
    cmap['euc-knn10'] = colors['color_green']
    cmap['euc-knn30'] = colors['color_darkgreen']
    cmap['apf-knn1'] = colors['color_lightblue']
    cmap['apf-knn10'] = colors['color_orange']
    cmap['apf-knn30'] = colors['color_yel']
    return cmap

def get_dash_map_models():
    # dashes=(dash_len, spacing)
    dash_map = {}
    dash_map['reference'] = (10, 0)
    dash_map['streamrak'] = (2, 2)
    dash_map['falkon'] = (4, 2)
    dash_map['euc-knn1']  = (3, 2, 1, 2, 1, 2)
    dash_map['euc-knn10'] = (3, 1, 1, 1, 1, 1)
    dash_map['euc-knn30'] = (10, 2, 2, 2, 2, 2)
    dash_map['apf-knn1'] = (5, 1, 3, 1)
    dash_map['apf-knn10'] = (10, 1)
    dash_map['apf-knn30'] = (20, 2)
    return dash_map

def get_marker_map_models():
    marker_map = {}
    marker_map['Reference perturbations'] = 'o'
    marker_map['reference'] = None
    marker_map['streamrak'] = 'o'
    marker_map['falkon'] = 's'
    marker_map['euc-knn1']  = 'v'
    marker_map['euc-knn10'] = 'p'
    marker_map['euc-knn30'] = 'd'
    marker_map['apf-knn1'] = 'h'
    marker_map['apf-knn10'] = 'D'
    marker_map['apf-knn30'] = 'P'
    return marker_map


def get_algo_names_map():
    algo_name_map = {}
    algo_name_map['Reference perturbations'] = 'Reference perturbations'
    algo_name_map['reference'] = 'Reference'
    algo_name_map['streamrak'] = 'Streamrak'
    algo_name_map['falkon'] = 'Falkon'
    algo_name_map['euc-knn1'] = 'Eucl-1-nn'
    algo_name_map['euc-knn10'] = 'Eucl-10-nn'
    algo_name_map['euc-knn30'] = 'Eucl-30-nn'
    algo_name_map['apf-knn1'] = 'Apf-1-nn'
    algo_name_map['apf-knn10'] = 'Apf-10-nn'
    algo_name_map['apf-knn30'] = 'Apf-30-nn'
    return algo_name_map