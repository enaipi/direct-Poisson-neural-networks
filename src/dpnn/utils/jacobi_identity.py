"""
Jacobi Identity Computation and Loss Evaluators.

Provides functional implementations of Jacobi identity loss variants for Poisson structures:
- exact (forward-mode AD)
- spectral (power iteration spectral norm)
- hutchinson / hutchinson_batch (stochastic trace estimators)
- manual (einsum with explicit Jacobian)
- exact_backward (backward-mode AD)

These functions accept a structure field callable L_func: z -> L(z) and a state batch z.
They can be used interchangeably during model training and postprocessing analysis.
"""

from typing import Callable, Optional
import torch
from torch import einsum


def compute_jacobi_loss(
    L_func: Callable[[torch.Tensor], torch.Tensor],
    z_n: torch.Tensor,
    mode: str = "spectral",
    num_samples: int = 10,
    get_jacobian_func: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> torch.Tensor:
    """
    Compute Jacobi identity loss for a given Poisson structure field L(z).

    Args:
        L_func: Callable taking z of shape (B, dim) and returning L(z) of shape (B, dim, dim).
        z_n: State tensor of shape (B, dim).
        mode: One of "exact", "exact_backward", "hutchinson", "hutchinson_batch", "spectral", "manual".
        num_samples: Number of random samples for Hutchinson / spectral iterations.
        get_jacobian_func: Optional callable for "manual" mode returning (B, dim, dim, dim).

    Returns:
        Scalar tensor containing the Jacobi loss.
    """
    if mode == "exact":
        return jacobi_loss_forward(L_func, z_n)
    elif mode == "exact_backward":
        return jacobi_loss_og(L_func, z_n)
    elif mode == "hutchinson":
        return jacobi_loss_hutchinson(L_func, z_n, num_samples=num_samples)
    elif mode == "hutchinson_batch":
        return jacobi_loss_hutchinson_batched(L_func, z_n, num_samples=num_samples)
    elif mode == "spectral":
        return jacobi_loss_spectral(L_func, z_n, num_iterations=num_samples)
    elif mode == "manual":
        return jacobi_loss_manual(L_func, z_n, get_jacobian_func=get_jacobian_func)
    else:
        raise ValueError(f"Unknown jacobi_loss_mode: '{mode}'")


def jacobi_loss_forward(
    L_func: Callable[[torch.Tensor], torch.Tensor],
    z_n: torch.Tensor
) -> torch.Tensor:
    """
    Exact squared Frobenius norm of Jacobiator using forward-mode AD.
    Fastest for small dimensions (dim < 100).
    """
    z_detached = z_n.detach()

    def compute_single_L(z_single):
        return L_func(z_single.unsqueeze(0)).squeeze(0)

    # Compute Jacobian of L with respect to z using forward-mode
    batch_jac = torch.func.vmap(torch.func.jacfwd(compute_single_L))(z_detached)

    # Compute L(z)
    Lz = L_func(z_n)

    # Compute cyclic terms of Jacobi tensor
    term1 = torch.einsum('bil,bjkl->bijk', Lz, batch_jac)
    term2 = term1.permute(0, 2, 3, 1)
    term3 = term1.permute(0, 3, 1, 2)

    return (term1 + term2 + term3).pow(2).mean()


def jacobi_loss_hutchinson(
    L_func: Callable[[torch.Tensor], torch.Tensor],
    z_n: torch.Tensor,
    num_samples: int = 10
) -> torch.Tensor:
    """
    Hutchinson trace estimator for Jacobi identity error.
    Memory-efficient for high dimensions.
    """
    B, dim = z_n.shape
    estimate = torch.zeros((), device=z_n.device, dtype=z_n.dtype)

    z_detached = z_n.detach().requires_grad_(True)
    Lz = L_func(z_detached)

    for i in range(num_samples):
        # Rademacher random vectors
        u = torch.randint(0, 2, (B, dim), device=z_n.device).float() * 2 - 1
        v = torch.randint(0, 2, (B, dim), device=z_n.device).float() * 2 - 1
        w = torch.randint(0, 2, (B, dim), device=z_n.device).float() * 2 - 1

        def compute_term_vec(vec_a, vec_b, vec_c):
            S = torch.einsum('bi,bij,bj->b', vec_b, Lz, vec_c)
            La = torch.einsum('bij,bj->bi', Lz, vec_a)
            grad_S = torch.autograd.grad(S.sum(), z_detached, create_graph=True,
                                        retain_graph=True)[0]
            return (La * grad_S).sum(dim=1)

        term1 = compute_term_vec(u, v, w)
        term2 = compute_term_vec(v, w, u)
        term3 = compute_term_vec(w, u, v)

        jacobi_v = (term1 + term2 + term3).pow(2)
        estimate = estimate + jacobi_v.mean()

    return estimate / num_samples


def jacobi_loss_hutchinson_batched(
    L_func: Callable[[torch.Tensor], torch.Tensor],
    z_n: torch.Tensor,
    num_samples: int = 10
) -> torch.Tensor:
    """
    Hutchinson estimator with all samples computed in parallel.
    More efficient for lower dimensions.
    """
    B, dim = z_n.shape
    total_samples = num_samples * B

    z_exp = z_n.repeat(num_samples, 1).detach().requires_grad_(True)
    L_exp = L_func(z_exp)

    u = torch.randint(0, 2, (total_samples, dim), device=z_n.device).float() * 2 - 1
    v = torch.randint(0, 2, (total_samples, dim), device=z_n.device).float() * 2 - 1
    w = torch.randint(0, 2, (total_samples, dim), device=z_n.device).float() * 2 - 1

    def compute_term_vec(vec_a, vec_b, vec_c, retain_graph=True):
        S = torch.einsum('bi,bij,bj->b', vec_b, L_exp, vec_c)
        La = torch.einsum('bij,bj->bi', L_exp, vec_a)
        grad_S = torch.autograd.grad(S.sum(), z_exp, create_graph=True,
                                    retain_graph=retain_graph)[0]
        return (La * grad_S).sum(dim=1)

    term1 = compute_term_vec(u, v, w, retain_graph=True)
    term2 = compute_term_vec(v, w, u, retain_graph=True)
    term3 = compute_term_vec(w, u, v, retain_graph=False)

    loss_i = (term1 + term2 + term3).pow(2)

    return loss_i.mean()


def jacobi_loss_spectral(
    L_func: Callable[[torch.Tensor], torch.Tensor],
    z_n: torch.Tensor,
    num_iterations: int = 10
) -> torch.Tensor:
    """
    Iterative spectral norm approximation using power iteration.
    Captures largest magnitude Jacobi violations.
    """
    B, dim = z_n.shape

    z_detached = z_n.detach().requires_grad_(True)
    Lz = L_func(z_detached)

    # Initialize random vectors
    u = torch.nn.functional.normalize(
        torch.randn(B, dim, device=z_n.device, dtype=z_n.dtype), dim=1
    )
    v = torch.nn.functional.normalize(
        torch.randn(B, dim, device=z_n.device, dtype=z_n.dtype), dim=1
    )
    w = torch.nn.functional.normalize(
        torch.randn(B, dim, device=z_n.device, dtype=z_n.dtype), dim=1
    )

    def get_jacobiator_scalar(u_vec, v_vec, w_vec, create_graph=False):
        def cyclic_term(a, b, c):
            S = torch.einsum('bi,bij,bj->b', b, Lz, c)
            grad_S = torch.autograd.grad(S.sum(), z_detached, create_graph=create_graph,
                                        retain_graph=True)[0]
            aL = torch.einsum('bi,bij->bj', a, Lz)
            return (aL * grad_S).sum(dim=1)

        return cyclic_term(u_vec, v_vec, w_vec) + \
               cyclic_term(v_vec, w_vec, u_vec) + \
               cyclic_term(w_vec, u_vec, v_vec)

    # Power iteration
    for i in range(num_iterations):
        u.requires_grad_(True)
        J_u = get_jacobiator_scalar(u, v, w, create_graph=True)
        grad_u = torch.autograd.grad(J_u.sum(), u)[0]
        u = torch.nn.functional.normalize(grad_u.detach(), dim=1)

        if i < num_iterations - 1:
            v.requires_grad_(True)
            J_v = get_jacobiator_scalar(u, v, w, create_graph=True)
            grad_v = torch.autograd.grad(J_v.sum(), v)[0]
            v = torch.nn.functional.normalize(grad_v.detach(), dim=1)

            w.requires_grad_(True)
            J_w = get_jacobiator_scalar(u, v, w, create_graph=True)
            grad_w = torch.autograd.grad(J_w.sum(), w)[0]
            w = torch.nn.functional.normalize(grad_w.detach(), dim=1)

    final_violation = get_jacobiator_scalar(u, v, w, create_graph=True)

    return (final_violation ** 2).mean()


def jacobi_loss_manual(
    L_func: Callable[[torch.Tensor], torch.Tensor],
    z_n: torch.Tensor,
    get_jacobian_func: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
) -> torch.Tensor:
    """
    Manual computation of Jacobi identity error using einsum.
    Slower but transparent - useful for debugging.
    """
    if get_jacobian_func is None:
        raise ValueError("manual mode requires get_jacobian_func parameter")

    Lz = L_func(z_n)
    J = get_jacobian_func(z_n)

    term1 = einsum('mkl,mijk->mijl', Lz, J)
    term2 = term1.permute(0, 2, 3, 1)
    term3 = term1.permute(0, 3, 1, 2)

    jacobi_identity_error = term1 + term2 + term3
    return jacobi_identity_error.pow(2).mean()


def jacobi_loss_og(
    L_func: Callable[[torch.Tensor], torch.Tensor],
    z_n: torch.Tensor
) -> torch.Tensor:
    """
    Original implementation using backward-mode functional.jacobian.
    Kept for compatibility and reference.
    """
    Lz = L_func(z_n)
    reduced_L = lambda z: torch.sum(L_func(z), axis=0)
    Lz_grad = torch.autograd.functional.jacobian(reduced_L, z_n, create_graph=True)\
              .permute(2, 0, 1, 3)

    term1 = einsum('mkl,mijk->mijl', Lz, Lz_grad)
    term2 = term1.permute(0, 2, 3, 1)
    term3 = term1.permute(0, 3, 1, 2)

    return (term1 + term2 + term3).pow(2).mean()
