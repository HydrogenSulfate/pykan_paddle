import paddle
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from .LBFGS import LBFGS
seed = 0
paddle.seed(seed=seed)


class MLP(paddle.nn.Layer):

    def __init__(self, width, act='identity', save_act=True, seed=0, device
        ='cpu'):
        super(MLP, self).__init__()
        paddle.seed(seed=seed)
        linears = []
        self.width = width
        self.depth = depth = len(width) - 1
        for i in range(depth):
            linears.append(paddle.nn.Linear(in_features=width[i],
                out_features=width[i + 1]))
        self.linears = paddle.nn.LayerList(sublayers=linears)
        if act == 'silu':
            self.act_fun = paddle.nn.Silu()
        elif act == 'relu':
            self.act_fun = paddle.nn.ReLU()
        elif act == 'identity':
            self.act_fun = paddle.nn.Identity()
        self.save_act = save_act
        self.acts = None
        self.device = device

    def get_act(self, x):
        if isinstance(x, dict):
            x = x['train_input']
        if x == None:
            if self.cache_data != None:
                x = self.cache_data
            else:
                raise Exception('missing input data x')
        save_act = self.save_act
        self.save_act = True
        self.forward(x)
        self.save_act = save_act

    @property
    def w(self):
        return [self.linears[l].weight for l in range(self.depth)]

    def forward(self, x):
        self.acts = []
        self.acts_scale = []
        self.wa_forward = []
        self.a_forward = []
        for i in range(self.depth):
            if self.save_act:
                act = x.clone()
                act_scale = paddle.std(x=x, axis=0)
                wa_forward = act_scale[None, :] * self.linears[i].weight
                self.acts.append(act)
                if i > 0:
                    self.acts_scale.append(act_scale)
                self.wa_forward.append(wa_forward)
            x = self.linears[i](x)
            if i < self.depth - 1:
                x = self.act_fun(x)
            elif self.save_act:
                act_scale = paddle.std(x=x, axis=0)
                self.acts_scale.append(act_scale)
        return x

    def attribute(self):
        if self.acts == None:
            self.get_act()
        node_scores = []
        edge_scores = []
        out_0 = paddle.ones(shape=self.width[-1])
        out_0.stop_gradient = not True
        node_score = out_0
        node_scores.append(node_score)
        for l in range(self.depth, 0, -1):
            edge_score = paddle.einsum('ij,i->ij', paddle.abs(x=self.
                wa_forward[l - 1]), node_score / (self.acts_scale[l - 1] + 
                0.0001))
            edge_scores.append(edge_score)
            node_score = paddle.sum(x=edge_score, axis=0) / paddle.sqrt(x=
                paddle.to_tensor(data=self.width[l - 1]))
            node_scores.append(node_score)
        self.node_scores = list(reversed(node_scores))
        self.edge_scores = list(reversed(edge_scores))
        self.wa_backward = self.edge_scores

    def plot(self, beta=3, scale=1.0, metric='w'):
        if metric == 'fa':
            self.attribute()
        depth = self.depth
        y0 = 0.5
        fig, ax = plt.subplots(figsize=(3 * scale, 3 * y0 * depth * scale))
        shp = self.width
        min_spacing = 1 / max(self.width)
        for j in range(len(shp)):
            N = shp[j]
            for i in range(N):
                plt.scatter(1 / (2 * N) + i / N, j * y0, s=min_spacing ** 2 *
                    5000 * scale ** 2, color='black')
        plt.ylim(-0.1 * y0, y0 * depth + 0.1 * y0)
        plt.xlim(-0.02, 1.02)
        linears = self.linears
        for ii in range(len(linears)):
            linear = linears[ii]
            p = linear.weight
            p_shp = tuple(p.shape)
            if metric == 'w':
                pass
            elif metric == 'act':
                p = self.wa_forward[ii]
            elif metric == 'fa':
                p = self.wa_backward[ii]
            else:
                raise Exception(
                    "metric = '{}' not recognized. Choices are 'w', 'act', 'fa'."
                    .format(metric))
            for i in range(p_shp[0]):
                for j in range(p_shp[1]):
                    plt.plot([1 / (2 * p_shp[0]) + i / p_shp[0], 1 / (2 *
                        p_shp[1]) + j / p_shp[1]], [y0 * (ii + 1), y0 * ii],
                        lw=0.5 * scale, alpha=np.tanh(beta * np.abs(p[i, j]
                        .detach().numpy())), color='blue' if p[i, j] > 0 else
                        'red')
        ax.axis('off')

    def reg(self, reg_metric, lamb_l1, lamb_entropy):
        if reg_metric == 'w':
            acts_scale = self.w
        if reg_metric == 'act':
            acts_scale = self.wa_forward
        if reg_metric == 'fa':
            acts_scale = self.wa_backward
        if reg_metric == 'a':
            acts_scale = self.acts_scale
        if len(tuple(acts_scale[0].shape)) == 2:
            reg_ = 0.0
            for i in range(len(acts_scale)):
                vec = acts_scale[i]
                vec = paddle.abs(x=vec)
                l1 = paddle.sum(x=vec)
                p_row = vec / (paddle.sum(x=vec, axis=1, keepdim=True) + 1)
                p_col = vec / (paddle.sum(x=vec, axis=0, keepdim=True) + 1)
                entropy_row = -paddle.mean(x=paddle.sum(x=p_row * paddle.
                    log2(x=p_row + 0.0001), axis=1))
                entropy_col = -paddle.mean(x=paddle.sum(x=p_col * paddle.
                    log2(x=p_col + 0.0001), axis=0))
                reg_ += lamb_l1 * l1 + lamb_entropy * (entropy_row +
                    entropy_col)
        elif len(tuple(acts_scale[0].shape)) == 1:
            reg_ = 0.0
            for i in range(len(acts_scale)):
                vec = acts_scale[i]
                vec = paddle.abs(x=vec)
                l1 = paddle.sum(x=vec)
                p = vec / (paddle.sum(x=vec) + 1)
                entropy = -paddle.sum(x=p * paddle.log2(x=p + 0.0001))
                reg_ += lamb_l1 * l1 + lamb_entropy * entropy
        return reg_

    def get_reg(self, reg_metric, lamb_l1, lamb_entropy):
        return self.reg(reg_metric, lamb_l1, lamb_entropy)

    def fit(self, dataset, opt='LBFGS', steps=100, log=1, lamb=0.0, lamb_l1
        =1.0, lamb_entropy=2.0, loss_fn=None, lr=1.0, batch=-1, metrics=
        None, in_vars=None, out_vars=None, beta=3, device='cpu', reg_metric
        ='w', display_metrics=None):
        if lamb > 0.0 and not self.save_act:
            print('setting lamb=0. If you want to set lamb > 0, set =True')
        old_save_act = self.save_act
        if lamb == 0.0:
            self.save_act = False
        pbar = tqdm(range(steps), desc='description', ncols=100)
        if loss_fn == None:
            loss_fn = loss_fn_eval = lambda x, y: paddle.mean(x=(x - y) ** 2)
        else:
            loss_fn = loss_fn_eval = loss_fn
        if opt == 'Adam':
            optimizer = paddle.optimizer.Adam(parameters=self.parameters(),
                learning_rate=lr, weight_decay=0.0)
        elif opt == 'LBFGS':
            optimizer = LBFGS(self.parameters(), lr=lr, history_size=10,
                line_search_fn='strong_wolfe', tolerance_grad=1e-32,
                tolerance_change=1e-32, tolerance_ys=1e-32)
        results = {}
        results['train_loss'] = []
        results['test_loss'] = []
        results['reg'] = []
        if metrics != None:
            for i in range(len(metrics)):
                results[metrics[i].__name__] = []
        if batch == -1 or batch > tuple(dataset['train_input'].shape)[0]:
            batch_size = tuple(dataset['train_input'].shape)[0]
            batch_size_test = tuple(dataset['test_input'].shape)[0]
        else:
            batch_size = batch
            batch_size_test = batch
        global train_loss, reg_

        def closure():
            global train_loss, reg_
            optimizer.clear_gradients(set_to_zero=False)
            pred = self.forward(dataset['train_input'][train_id].to(self.
                device))
            train_loss = loss_fn(pred, dataset['train_label'][train_id].to(
                self.device))
            if self.save_act:
                if reg_metric == 'fa':
                    self.attribute()
                reg_ = self.get_reg(reg_metric, lamb_l1, lamb_entropy)
            else:
                reg_ = paddle.to_tensor(data=0.0)
            objective = train_loss + lamb * reg_
            objective.backward()
            return objective
        for _ in pbar:
            if _ == steps - 1 and old_save_act:
                self.save_act = True
            train_id = np.random.choice(tuple(dataset['train_input'].shape)
                [0], batch_size, replace=False)
            test_id = np.random.choice(tuple(dataset['test_input'].shape)[0
                ], batch_size_test, replace=False)
            if opt == 'LBFGS':
                optimizer.step(closure)
            if opt == 'Adam':
                pred = self.forward(dataset['train_input'][train_id].to(
                    self.device))
                train_loss = loss_fn(pred, dataset['train_label'][train_id]
                    .to(self.device))
                if self.save_act:
                    reg_ = self.get_reg(reg_metric, lamb_l1, lamb_entropy)
                else:
                    reg_ = paddle.to_tensor(data=0.0)
                loss = train_loss + lamb * reg_
                optimizer.clear_gradients(set_to_zero=False)
                loss.backward()
                optimizer.step()
            test_loss = loss_fn_eval(self.forward(dataset['test_input'][
                test_id].to(self.device)), dataset['test_label'][test_id].
                to(self.device))
            if metrics != None:
                for i in range(len(metrics)):
                    results[metrics[i].__name__].append(metrics[i]().item())
            results['train_loss'].append(paddle.sqrt(x=train_loss).cpu().
                detach().numpy())
            results['test_loss'].append(paddle.sqrt(x=test_loss).cpu().
                detach().numpy())
            results['reg'].append(reg_.cpu().detach().numpy())
            if _ % log == 0:
                if display_metrics == None:
                    pbar.set_description(
                        '| train_loss: %.2e | test_loss: %.2e | reg: %.2e | ' %
                        (paddle.sqrt(x=train_loss).cpu().detach().numpy(),
                        paddle.sqrt(x=test_loss).cpu().detach().numpy(),
                        reg_.cpu().detach().numpy()))
                else:
                    string = ''
                    data = ()
                    for metric in display_metrics:
                        string += f' {metric}: %.2e |'
                        try:
                            results[metric]
                        except:
                            raise Exception(f'{metric} not recognized')
                        data += results[metric][-1],
                    pbar.set_description(string % data)
        return results
