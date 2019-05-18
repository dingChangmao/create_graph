import re
import uuid
import Drawing_flow_chart as gv
import pyautogui
import time

#   改配置项为图像节点的颜色形状等等
graph_pref = {
    'fontcolor': '#414141',
    'style': 'rounded',
}

name_scope_graph_pref = {
    'bgcolor': '#eeeeee',
    'color': '#aaaaaa',
    'penwidth': '2',
}

non_name_scope_graph_pref = {
    'fillcolor':  'white',
    'color': 'white',
}

node_pref = {
    'style': 'filled',
    'fillcolor': 'white',
    'color': '#aaaaaa',
    'penwidth': '2',
    'fontcolor': '#414141',
}
# 改变边的颜色字体等等
edge_pref = {
    'color': '#aaaaaa',  # 白色
    'arrowsize': '1.2',
    'penwidth': '2.5',
    'fontcolor': '#414141',   # 灰色
}
# index of subgraph
# 给子图取名  默认0+
# 随迭代加一
CLUSTER_INDEX = 0



def tf_digraph(name=None, name_scope=None, style=True):
   #  决定输出的文件名字，格式，以及画图
    digraph = gv.Digraph(name=name)

    # 默认输出为G.gv.png
    digraph.format = 'png'

    if name_scope:
        # 标签名用于节点的展示  理论上tf越完整图形越美观
        digraph.graph_attr['label'] = name_scope

    if style is False: return digraph

    if name_scope:

        digraph.graph_attr.update(name_scope_graph_pref)
    else:
        digraph.graph_attr.update(non_name_scope_graph_pref)
    digraph.graph_attr.update(graph_pref)
    digraph.node_attr.update(node_pref)
    digraph.edge_attr.update(edge_pref)
    return digraph


def nested_dict(dict_, keys, val):
        #  生成嵌套的字典
    cloned = dict_.copy()
    if len(keys) == 1:
        cloned[keys[0]] = val
        return cloned
    dd = cloned[keys[0]]

    for k in keys[1:len(keys)-1]:
        dd = dd[k]
    last_key = keys[len(keys)-1]
    dd[last_key] = val

    return cloned


def node_abs_paths(node):
    # 节点
    node_names = node.name.split('/')
    return ['/'.join(node_names[0:i+1]) for i in range(len(node_names))]



def node_table(tfgraph, depth=1):
    # 节点展示
    table = {}
    max_depth = depth
    ops = tfgraph.get_operations()
    # 从tf中获取节点信息
    # 数据
    for depth_i in range(max_depth):

        for op in ops:
            abs_paths = node_abs_paths(op)
            if depth_i >= len(abs_paths): continue
            ps = abs_paths[:depth_i+1]
            if len(ps) == 1:
                key = '/'.join(abs_paths[0:depth_i+1])
                if not key in table: table[key] = {}
            else:
                table = nested_dict(table, ps, {})

    return table


def node_shape(tfnode, depth=1):
    # 节点形状
    outpt_name = tfnode.name

    if len(outpt_name.split('/')) < depth: return None

    on = '/'.join(outpt_name.split('/')[:depth]) # output node

    result = re.match(r"(.*):\d*$", on)

    if not result: return None
    on = result.groups()[0]
    if tfnode.shape.ndims is None:
        return on, []
    else:
        return on, tfnode.shape.as_list()


def node_input_table(tfgraph, depth=1):
    # 节点的输入

    table = {}
    inpt_op_table = {}
    inpt_op_shape_table = {}
    for op in tfgraph.get_operations():
        op_name = op.name.split('/')[0:depth]
        opn = '/'.join(op_name)

        if not opn in inpt_op_table:
            inpt_op_table[opn] = []
        inpt_op_list = ['/'.join(inpt_op.split('/')[0:depth]) for inpt_op in op.node_def.input]
        inpt_op_table[opn].append(inpt_op_list)

        for output in op.outputs:
            for i in range(depth):
                shape = node_shape(output, depth=i+1)
                if shape: inpt_op_shape_table[shape[0]] = shape[1]

    for opn in inpt_op_table.keys():
        t_l = []
        for ll in inpt_op_table[opn]:
            list.extend(t_l, ll)
        table[opn] = list(set(t_l))
    return table, inpt_op_shape_table


def add_nodes(node_table, name=None, name_scope=None, style=True):
    # 添加节点
    global CLUSTER_INDEX
    if name:
        digraph = tf_digraph(name=name, name_scope=name_scope, style=style)
    else:
        digraph = tf_digraph(name=str(uuid.uuid4().get_hex().upper()[0:6]), name_scope=name_scope, style=style)
    graphs = []
    for key, value in node_table.items():
        if len(value) > 0:
            sg = add_nodes(value, name='cluster_%i' % CLUSTER_INDEX, name_scope=key.split('/')[-1], style=style)
            sg.node(key, key.split('/')[-1])
            CLUSTER_INDEX += 1
            graphs.append(sg)
        else:
            digraph.node(key, key.split('/')[-1])
    for tg in graphs:
        digraph.subgraph(tg)
    return digraph


def edge_label(shape):
    # 边的数据
    if len(shape) == 0: return ''
    if shape[0] is None: label = "?"
    else: label = "%i" % shape[0]
    for s in shape[1:]:
        if s is None: label += "×?"
        else: label += u"×%i" % s
    return label


def add_edges(digraph, node_inpt_table, node_inpt_shape_table):
    # 添加边（节点与节点的关系）
    for node, node_inputs in node_inpt_table.items():
        if re.match(r"\^", node): continue
        for ni in node_inputs:
            if ni == node: continue
            if re.match(r"\^", ni): continue
            if not ni in node_inpt_shape_table:
                digraph.edge(ni, node)
            else:
                shape = node_inpt_shape_table[ni]
                digraph.edge(ni, node, label=edge_label(shape))
    return digraph


def board(tfgraph, depth=1, name='G', style=True):
    # main
    # board(tf.get_default_graph(),depth=2).render(view = False)
    #  view  默认为Ture  运行时会展示图像，后台运行时建议设置false

    global CLUSTER_INDEX

    CLUSTER_INDEX = 0

    _node_table = node_table(tfgraph, depth=depth) # 放入tfgraph对象
    _node_inpt_table, _node_inpt_shape_table = node_input_table(tfgraph, depth=depth)
    digraph = add_nodes(_node_table, name=name, style=style)
    digraph = add_edges(digraph, _node_inpt_table, _node_inpt_shape_table)

    return digraph

def run_demo(tfgraph,depth=1, name='G', style=True):
    # 通过模拟键盘，在不影响模型训练的过程下先获取流程图
    g = board(tfgraph).render(filename='graph', view=True)
    # g = board(tf.get_default_graph()).save()
    # pyautogui.press('esc')
    pyautogui.keyDown('escape')
    time.sleep(2)
    pyautogui.keyUp('escape')
    return 'ok'



