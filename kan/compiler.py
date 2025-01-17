import paddle
from sympy import *
import sympy
import numpy as np
from kan.MultKAN import MultKAN


def next_nontrivial_operation(expr, scale=1, bias=0):
    if expr.func == Add or expr.func == Mul:
        n_arg = len(expr.args)
        n_num = 0
        n_var_id = []
        n_num_id = []
        var_args = []
        for i in range(n_arg):
            is_number = expr.args[i].is_number
            n_num += is_number
            if not is_number:
                n_var_id.append(i)
                var_args.append(expr.args[i])
            else:
                n_num_id.append(i)
        if n_num > 0:
            if expr.func == Add:
                for i in range(n_num):
                    if i == 0:
                        bias = expr.args[n_num_id[i]]
                    else:
                        bias += expr.args[n_num_id[i]]
            if expr.func == Mul:
                for i in range(n_num):
                    if i == 0:
                        scale = expr.args[n_num_id[i]]
                    else:
                        scale *= expr.args[n_num_id[i]]
            return next_nontrivial_operation(expr.func(*var_args), scale, bias)
        else:
            return expr, scale, bias
    else:
        return expr, scale, bias


def expr2kan(input_variables, expr, grid=5, k=3, auto_save=True):


    class Node:

        def __init__(self, expr, mult_bool, depth, scale, bias, parent=None,
            mult_arity=None):
            self.expr = expr
            self.mult_bool = mult_bool
            if self.mult_bool:
                self.mult_arity = mult_arity
            self.depth = depth
            if len(Nodes) <= depth:
                Nodes.append([])
                index = 0
            else:
                index = len(Nodes[depth])
            Nodes[depth].append(self)
            self.index = index
            if parent == None:
                self.parent_index = None
            else:
                self.parent_index = parent.index
            self.child_index = []
            if parent != None:
                parent.child_index.append(self.index)
            self.scale = scale
            self.bias = bias


    class SubNode:

        def __init__(self, expr, depth, scale, bias, parent=None):
            self.expr = expr
            self.depth = depth
            if len(SubNodes) <= depth:
                SubNodes.append([])
                index = 0
            else:
                index = len(SubNodes[depth])
            SubNodes[depth].append(self)
            self.index = index
            self.parent_index = None
            self.child_index = []
            parent.child_index.append(self.index)
            self.scale = scale
            self.bias = bias


    class Connection:

        def __init__(self, affine, fun, fun_name, parent=None, child=None,
            power_exponent=None):
            self.affine = affine
            self.fun = fun
            self.fun_name = fun_name
            self.parent_index = parent.index
            self.depth = parent.depth
            self.child_index = child.index
            self.power_exponent = power_exponent
            Connections[self.depth, self.parent_index, self.child_index] = self

    def create_node(expr, parent=None, n_layer=None):
        expr, scale, bias = next_nontrivial_operation(expr)
        if parent == None:
            depth = 0
        else:
            depth = parent.depth
        if expr.func == Mul:
            mult_arity = len(expr.args)
            node = Node(expr, True, depth, scale, bias, parent=parent,
                mult_arity=mult_arity)
            for i in range(mult_arity):
                expr_i, scale, bias = next_nontrivial_operation(expr.args[i])
                subnode = SubNode(expr_i, node.depth + 1, scale, bias,
                    parent=node)
                if expr_i.func == Add:
                    for j in range(len(expr_i.args)):
                        expr_ij, scale, bias = next_nontrivial_operation(expr_i
                            .args[j])
                        if expr_ij.func == Mul:
                            new_node = create_node(expr_ij, parent=subnode,
                                n_layer=n_layer)
                            c = Connection([1, 0, float(scale), float(bias)
                                ], lambda x: x, 'x', parent=subnode, child=
                                new_node)
                        elif expr_ij.func == Symbol:
                            new_node = create_node(expr_ij, parent=subnode,
                                n_layer=n_layer)
                            c = Connection([1, 0, float(scale), float(bias)
                                ], lambda x: x, fun_name='x', parent=
                                subnode, child=new_node)
                        else:
                            new_node = create_node(expr_ij.args[0], parent=
                                subnode, n_layer=n_layer)
                            if expr_ij.func == Pow:
                                power_exponent = expr_ij.args[1]
                            else:
                                power_exponent = None
                            Connection([1, 0, float(scale), float(bias)],
                                expr_ij.func, fun_name=expr_ij.func, parent
                                =subnode, child=new_node, power_exponent=
                                power_exponent)
                elif expr_i.func == Mul:
                    new_node = create_node(expr_i, parent=subnode, n_layer=
                        n_layer)
                    Connection([1, 0, 1, 0], lambda x: x, fun_name='x',
                        parent=subnode, child=new_node)
                elif expr_i.func == Symbol:
                    new_node = create_node(expr_i, parent=subnode, n_layer=
                        n_layer)
                    Connection([1, 0, 1, 0], lambda x: x, fun_name='x',
                        parent=subnode, child=new_node)
                else:
                    new_node = create_node(expr_i.args[0], parent=subnode,
                        n_layer=n_layer)
                    if expr_i.func == Pow:
                        power_exponent = expr_i.args[1]
                    else:
                        power_exponent = None
                    Connection([1, 0, 1, 0], expr_i.func, fun_name=expr_i.
                        func, parent=subnode, child=new_node,
                        power_exponent=power_exponent)
        elif expr.func == Add:
            node = Node(expr, False, depth, scale, bias, parent=parent)
            subnode = SubNode(expr, node.depth + 1, 1, 0, parent=node)
            for i in range(len(expr.args)):
                expr_i, scale, bias = next_nontrivial_operation(expr.args[i])
                if expr_i.func == Mul:
                    new_node = create_node(expr_i, parent=subnode, n_layer=
                        n_layer)
                    Connection([1, 0, float(scale), float(bias)], lambda x:
                        x, fun_name='x', parent=subnode, child=new_node)
                elif expr_i.func == Symbol:
                    new_node = create_node(expr_i, parent=subnode, n_layer=
                        n_layer)
                    Connection([1, 0, float(scale), float(bias)], lambda x:
                        x, fun_name='x', parent=subnode, child=new_node)
                else:
                    new_node = create_node(expr_i.args[0], parent=subnode,
                        n_layer=n_layer)
                    if expr_i.func == Pow:
                        power_exponent = expr_i.args[1]
                    else:
                        power_exponent = None
                    Connection([1, 0, float(scale), float(bias)], expr_i.
                        func, fun_name=expr_i.func, parent=subnode, child=
                        new_node, power_exponent=power_exponent)
        elif expr.func == Symbol:
            if n_layer == None:
                node = Node(expr, False, depth, scale, bias, parent=parent)
            else:
                node = Node(expr, False, depth, scale, bias, parent=parent)
                return_node = node
                for i in range(n_layer - depth):
                    subnode = SubNode(expr, node.depth + 1, 1, 0, parent=node)
                    node = Node(expr, False, subnode.depth, 1, 0, parent=
                        subnode)
                    Connection([1, 0, 1, 0], lambda x: x, fun_name='x',
                        parent=subnode, child=node)
                node = return_node
            Start_Nodes.append(node)
        else:
            node = Node(expr, False, depth, scale, bias, parent=parent)
            expr_i, scale, bias = next_nontrivial_operation(expr.args[0])
            subnode = SubNode(expr_i, node.depth + 1, 1, 0, parent=node)
            new_node = create_node(expr.args[0], parent=subnode, n_layer=
                n_layer)
            if expr.func == Pow:
                power_exponent = expr.args[1]
            else:
                power_exponent = None
            Connection([1, 0, 1, 0], expr.func, fun_name=expr.func, parent=
                subnode, child=new_node, power_exponent=power_exponent)
        return node
    Nodes = [[]]
    SubNodes = [[]]
    Connections = {}
    Start_Nodes = []
    create_node(expr, n_layer=None)
    n_layer = len(Nodes) - 1
    Nodes = [[]]
    SubNodes = [[]]
    Connections = {}
    Start_Nodes = []
    create_node(expr, n_layer=n_layer)
    for node in Start_Nodes:
        c = Connections[node.depth, node.parent_index, node.index]
        c.affine[0] = float(node.scale)
        c.affine[1] = float(node.bias)
        node.scale = 1.0
        node.bias = 0.0
    node2var = []
    for node in Start_Nodes:
        for i in range(len(input_variables)):
            if node.expr == input_variables[i]:
                node2var.append(i)
    n_mult = []
    n_sum = []
    for layer in Nodes:
        n_mult.append(0)
        n_sum.append(0)
        for node in layer:
            if node.mult_bool == True:
                n_mult[-1] += 1
            else:
                n_sum[-1] += 1
    n_layer = len(Nodes) - 1
    subnode_index_convert = {}
    node_index_convert = {}
    connection_index_convert = {}
    mult_arities = []
    for layer_id in range(n_layer + 1):
        mult_arity = []
        i_sum = 0
        i_mult = 0
        for i in range(len(Nodes[layer_id])):
            node = Nodes[layer_id][i]
            if node.mult_bool == True:
                kan_node_id = n_sum[layer_id] + i_mult
                arity = len(node.child_index)
                for i in range(arity):
                    subnode = SubNodes[node.depth + 1][node.child_index[i]]
                    kan_subnode_id = n_sum[layer_id] + np.sum(mult_arity) + i
                    subnode_index_convert[subnode.depth, subnode.index] = int(
                        n_layer - subnode.depth), int(kan_subnode_id)
                i_mult += 1
                mult_arity.append(arity)
            else:
                kan_node_id = i_sum
                if len(node.child_index) > 0:
                    subnode = SubNodes[node.depth + 1][node.child_index[0]]
                    kan_subnode_id = i_sum
                    subnode_index_convert[subnode.depth, subnode.index] = int(
                        n_layer - subnode.depth), int(kan_subnode_id)
                i_sum += 1
            if layer_id == n_layer:
                node_index_convert[node.depth, node.index] = int(n_layer -
                    node.depth), int(node2var[kan_node_id])
            else:
                node_index_convert[node.depth, node.index] = int(n_layer -
                    node.depth), int(kan_node_id)
        mult_arities.append(mult_arity)
    for index in list(Connections.keys()):
        depth, subnode_id, node_id = index
        _, kan_subnode_id = subnode_index_convert[depth, subnode_id]
        _, kan_node_id = node_index_convert[depth, node_id]
        connection_index_convert[depth, subnode_id, node_id
            ] = n_layer - depth, kan_subnode_id, kan_node_id
    n_sum.reverse()
    n_mult.reverse()
    mult_arities.reverse()
    width = [[n_sum[i], n_mult[i]] for i in range(len(n_sum))]
    width[0][0] = len(input_variables)
    model = MultKAN(width=width, mult_arity=mult_arities, grid=grid, k=k,
        auto_save=False)
    for l in range(model.depth):
        for i in range(model.width_in[l]):
            for j in range(model.width_out[l + 1]):
                model.fix_symbolic(l, i, j, '0', fit_params_bool=False)
    Nodes_flat = [x for xs in Nodes for x in xs]
    self = model
    for node in Nodes_flat:
        node_depth = node.depth
        node_index = node.index
        kan_node_depth, kan_node_index = node_index_convert[node_depth,
            node_index]
        if kan_node_depth > 0:
            self.node_scale[kan_node_depth - 1].detach()[kan_node_index
                ] = float(node.scale)
            self.node_bias[kan_node_depth - 1].detach()[kan_node_index
                ] = float(node.bias)
    SubNodes_flat = [x for xs in SubNodes for x in xs]
    for subnode in SubNodes_flat:
        subnode_depth = subnode.depth
        subnode_index = subnode.index
        kan_subnode_depth, kan_subnode_index = subnode_index_convert[
            subnode_depth, subnode_index]
        self.subnode_scale[kan_subnode_depth].detach()[kan_subnode_index
            ] = float(subnode.scale)
        self.subnode_bias[kan_subnode_depth].detach()[kan_subnode_index
            ] = float(subnode.bias)
    Connections_flat = list(Connections.values())
    for connection in Connections_flat:
        c_depth = connection.depth
        c_j = connection.parent_index
        c_i = connection.child_index
        kc_depth, kc_j, kc_i = connection_index_convert[c_depth, c_j, c_i]
        fun_name = connection.fun_name
        if fun_name == 'x':
            kfun_name = 'x'
        elif fun_name == exp:
            kfun_name = 'exp'
        elif fun_name == sin:
            kfun_name = 'sin'
        elif fun_name == cos:
            kfun_name = 'cos'
        elif fun_name == tan:
            kfun_name = 'tan'
        elif fun_name == sqrt:
            kfun_name = 'sqrt'
        elif fun_name == log:
            kfun_name = 'log'
        elif fun_name == tanh:
            kfun_name = 'tanh'
        elif fun_name == asin:
            kfun_name = 'arcsin'
        elif fun_name == acos:
            kfun_name = 'arccos'
        elif fun_name == atan:
            kfun_name = 'arctan'
        elif fun_name == atanh:
            kfun_name = 'arctanh'
        elif fun_name == sign:
            kfun_name = 'sgn'
        elif fun_name == Pow:
            alpha = connection.power_exponent
            if alpha == Rational(1, 2):
                kfun_name = 'x^0.5'
            elif alpha == -Rational(1, 2):
                kfun_name = '1/x^0.5'
            elif alpha == Rational(3, 2):
                kfun_name = 'x^1.5'
            else:
                alpha = int(connection.power_exponent)
                if alpha > 0:
                    if alpha == 1:
                        kfun_name = 'x'
                    else:
                        kfun_name = f'x^{alpha}'
                elif alpha == -1:
                    kfun_name = '1/x'
                else:
                    kfun_name = f'1/x^{-alpha}'
        model.fix_symbolic(kc_depth, kc_i, kc_j, kfun_name, fit_params_bool
            =False)
        model.symbolic_fun[kc_depth].affine.data.reshape(self.width_out[
            kc_depth + 1], self.width_in[kc_depth], 4)[kc_j][kc_i
            ] = paddle.to_tensor(data=connection.affine)
    model.auto_save = auto_save
    model.log_history('kanpiler')
    return model


kanpiler = expr2kan
