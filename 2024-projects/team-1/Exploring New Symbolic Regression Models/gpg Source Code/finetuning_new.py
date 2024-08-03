from pygpg import conversion as C
import numpy as np
import torch
import sympy
from copy import deepcopy
import re
from sklearn.metrics import silhouette_score
import torch.nn as nn

"""
Fine-tunes a sympy model. Returns the fine-tuned model and the number of steps used.
If it terminates prematurely, the number of steps used is returned as well.
"""


def finetune(sympy_model, X, y, label_indices=None, Z_vals=None, loss_method='regular', cluster_labels=None,
             weight=None, bounds=(1, 3.05, 9.15, 102), learning_rate=1.0,
             n_steps=100,
             tol_grad=1e-9, tol_change=1e-9):
    best_torch_model, best_loss = None, np.infty

    if not isinstance(X, torch.TensorType):
        X = torch.tensor(X)
    if not isinstance(y, torch.TensorType):
        y = torch.tensor(y.reshape((-1,)))

        # workaround to have identical constants be treated as different ones
    str_model = str(sympy_model)
    sympy_model = sympy.sympify(str(sympy_model))
    for el in sympy.preorder_traversal(sympy_model):
        if isinstance(el, sympy.Float):
            f = float(el)
            str_model = str_model.replace(str(f), str(f + np.random.normal(0, 1e-5)), 1)
    sympy_model = sympy.sympify(str_model)

    expr_vars = set(re.findall(r'\bx_[0-9]+', str(sympy_model)))
    try:
        torch_model = C.sympy_to_torch(sympy_model, timeout=5)
    except TypeError:
        print("[!] Warning: invalid conversion from sympy to torch pre fine-tuning")
        return sympy_model, 0
    if torch_model is None:
        print("[!] Warning: failed to convert from sympy to torch within a reasonable time")
        return sympy_model, 0

    x_args = {x: X[:, int(x.lstrip("x_"))] for x in expr_vars}

    try:  # optimizer might get an empty parameter list
        optimizer = torch.optim.LBFGS(
            torch_model.parameters(),
            line_search_fn=None,
            lr=learning_rate,
            tolerance_grad=tol_grad,
            tolerance_change=tol_change)
    except ValueError:
        return sympy_model, 0

    prev_loss = np.infty
    batch_x = x_args
    batch_y = y
    steps_done = 0
    for _ in range(n_steps):
        steps_done += 1
        optimizer.zero_grad()
        try:
            # predict rainfall rates
            p = torch_model(**batch_x).squeeze(-1)
        except TypeError as err:
            print("[!] Warning: error during forward call of torch model while fine-tuning")
            print(err)
            return sympy_model, steps_done
        # regular loss
        if loss_method == 'regular':
            loss = (p - batch_y).pow(2).mean().div(2)
        # binned rainfall loss
        if loss_method == 'binned-rainfall':
            relu0_0, relu0_1, relu1_0, relu1_1, relu2_0, relu2_1 = 0, 0, 0, 0, 0, 0
            if not torch.any(torch.isnan(p)) and (p.dim() != 0):
                # Separate the predicted rainfall rates into their groups and compute the ReLU terms
                p_0, p_1, p_2 = p[label_indices[0]], p[label_indices[1]], p[label_indices[2]]
                relu0_0 = torch.sum(torch.tensor([max(0, bounds[0] - pred) for pred in p_0]))
                relu0_1 = torch.sum(torch.tensor([max(0, pred - bounds[1]) for pred in p_0]))
                relu1_0 = torch.sum(torch.tensor([max(0, bounds[1] - pred) for pred in p_1]))
                relu1_1 = torch.sum(torch.tensor([max(0, pred - bounds[2]) for pred in p_1]))
                relu2_0 = torch.sum(torch.tensor([max(0, bounds[2] - pred) for pred in p_2]))
                relu2_1 = torch.sum(torch.tensor([max(0, pred - bounds[3]) for pred in p_2]))
            else:
                if _ == 0:
                    print(f'Predicted values contain N/A, set {loss_method} loss terms to 0.')
            # Define loss term
            loss = 1 * (p - batch_y).pow(2).mean().div(2) + weight * (
                    relu0_0 + relu0_1 + relu1_0 + relu1_1 + relu2_0 + relu2_1)
        # Cluster-based loss
        if loss_method == 'clusters':
            sil_score = 0
            if not torch.any(torch.isnan(p)) and (p.dim() != 0):
                # Compute silhouette score
                sil_score = silhouette_score(X=p.detach().numpy().reshape(-1, 1), labels=cluster_labels)
            else:
                if _ == 0:
                    print(f'Predicted values contain N/A, set {loss_method} loss terms to 0.')
            # Define loss term
            loss = 1*(p - batch_y).pow(2).mean().div(2) - weight * sil_score
        # Z-R loss
        if loss_method == 'Z-R':
            a = 134
            b = 1.6
            Z = torch.tensor(Z_vals.reshape(-1))
            # Define loss term
            loss = 1 * (p - batch_y).pow(2).mean().div(2) + weight * (p - (Z / a) ** (1 / b)).pow(2).mean()
        loss.retain_grad()
        loss.backward()
        optimizer.step(lambda: loss)
        loss_val = loss.item()
        if loss_val < best_loss:
            best_torch_model = deepcopy(torch_model)
            best_loss = loss_val
        if abs(loss_val - prev_loss) < tol_change:
            break
        prev_loss = loss_val
    result = best_torch_model.sympy()[0] if best_torch_model else sympy_model
    result = C.timed_simplify(result, timeout=5)
    if result is None:
        return sympy_model, steps_done
    return result, steps_done
