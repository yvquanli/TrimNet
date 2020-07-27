import random
import os
import torch
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr, kendalltau, pearsonr
from torch.nn import LayerNorm, BatchNorm1d
from torch_geometric.utils import remove_self_loops


class NodeLevelBatchNorm(BatchNorm1d):
    r"""
    Applies Batch Normalization over a batch of graph data.
    Shape:
        - Input: [batch_nodes_dim, node_feature_dim]
        - Output: [batch_nodes_dim, node_feature_dim]
    batch_nodes_dim: all nodes of a batach graph
    """

    def _check_input_dim(self, input):
        if input.dim() != 2:
            raise ValueError('expected 2D input (got {}D input)'
                             .format(input.dim()))

    def extra_repr(self):
        return '{num_features}, affine={affine}'.format(**self.__dict__)


class NodeLevelLayerNorm(LayerNorm):
    r"""
    Applies node level layer normalization over a batch of graph data.
    LayerNorm in/out: [N, **] number of examples, etc.
    Shape:
        - Input: [batch_nodes_dim, node_feature_dim]
        - Output: [batch_nodes_dim, node_feature_dim]
    """

    def _check_input_dim(self, input):
        if input.dim() != 2:
            raise ValueError('expected 2D input (got {}D input)'
                             .format(input.dim()))

    def extra_repr(self):
        return '{normalized_shape},' \
               'elementwise_affine={elementwise_affine}'.format(**self.__dict__)


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class Complete(object):
    def __call__(self, data):
        device = data.edge_index.device

        row = torch.arange(data.num_nodes, dtype=torch.long, device=device)
        col = torch.arange(data.num_nodes, dtype=torch.long, device=device)

        row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
        col = col.repeat(data.num_nodes)
        edge_index = torch.stack([row, col], dim=0)

        edge_attr = None
        if data.edge_attr is not None:
            idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]
            size = list(data.edge_attr.size())
            size[0] = data.num_nodes * data.num_nodes
            edge_attr = data.edge_attr.new_zeros(size)
            edge_attr[idx] = data.edge_attr

        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        data.edge_attr = edge_attr
        data.edge_index = edge_index

        return data


def metrics(y_true, y_pred, info='test'):  # true, pred np.float
    return {
        info + ' mse': mean_squared_error(y_true, y_pred),
        info + ' rmse': mean_squared_error(y_true, y_pred) ** 0.5,
        info + ' mae': mean_absolute_error(y_true, y_pred),
        info + ' r^2': r2_score(y_true, y_pred),
        info + ' pearson r': pearsonr(y_true.tolist(), y_pred.tolist())[0],
        info + ' spearman rho': spearmanr(y_true.tolist(), y_pred.tolist())[0],
        info + ' kendall tau': kendalltau(y_true.tolist(), y_pred.tolist())[0],
    }


def get_latest_ckpt(file_dir='./ckpt/'):
    filelist = os.listdir(file_dir)
    filelist.sort(key=lambda fn: os.path.getmtime(file_dir + fn) if not os.path.isdir(file_dir + fn) else 0)
    print('The latest ckpt is {}'.format(filelist[-1]))
    return file_dir + filelist[-1]


def angle(vector1, vector2):
    cos_angle = vector1.dot(vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    angle = np.arccos(cos_angle)
    # angle2=angle*360/2/np.pi
    return angle  # , angle2


def area_triangle(vector1, vector2):
    trianglearea = 0.5 * np.linalg.norm( \
        np.cross(vector1, vector2))
    return trianglearea


def area_triangle_vertex(vertex1, vertex2, vertex3):
    trianglearea = 0.5 * np.linalg.norm( \
        np.cross(vertex2 - vertex1, vertex3 - vertex1))
    return trianglearea


def cal_angle_area(vector1, vector2):
    return angle(vector1, vector2), area_triangle(vector1, vector2)


# vij=np.array([ 0, 1,  1])
# vik=np.array([ 0, 2,  0])
# cal_angle_area(vij, vik)   # (0.7853981633974484, 1.0)


def cal_dist(vertex1, vertex2, ord=2):
    return np.linalg.norm(vertex1 - vertex2, ord=ord)
# vertex1 = np.array([1,2,3])
# vertex2 = np.array([4,5,6])
# cal_dist(vertex1, vertex2, ord=1), np.sum(vertex1-vertex2) # (9.0, -9)
# cal_dist(vertex1, vertex2, ord=2), np.sqrt(np.sum(np.square(vertex1-vertex2)))  # (5.196152422706632, 5.196152422706632)
# cal_dist(vertex1, vertex2, ord=3)  # 4.3267487109222245
