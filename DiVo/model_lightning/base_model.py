import torch

def generate_mask_tensor(ori_shape:list, order:int, device='cuda') -> torch.Tensor:
    m, n = ori_shape
    M, N = m ** order, n ** order

    row_indices = torch.arange(M, device=device)
    col_indices = torch.arange(N, device=device)

    I, J = torch.meshgrid(row_indices, col_indices, indexing='ij')  # I.shape = J.shape = (M, N)

    def decompose_indices(indices, base, order):
        """将 flat index 转为 base 进制展开，返回 shape: (num, order)"""
        exps = torch.tensor([base ** i for i in reversed(range(order))], device=device)
        digits = (indices.unsqueeze(-1) // exps) % base
        return digits

    I_digits = decompose_indices(I.flatten(), m, order)
    J_digits = decompose_indices(J.flatten(), n, order)

    combos = torch.cat([I_digits, J_digits], dim=1)

    _, unique_indices = torch.unique(combos, dim=0, return_inverse=True)

    first_occurrence = torch.zeros_like(unique_indices, dtype=torch.bool)
    first_occurrence[unique_indices.flip(0).unique(return_inverse=False)] = True
    mask = first_occurrence.view(M, N)

    return mask


def kronecker_product(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Args  
    ---
        A (Tensor) : A Tensor with the shape of (batch_size, m, n)  
        B (Tensor) : B Tensor with the shape of (batch_size, p, q)  
    Return
    ---
        result (Tensor) : Kronecker product of A and B for each batch,   
                          with the shape of (batch_size, m*p, n*q)  
    """
    batch_size, m, n = A.shape
    _, p, q = B.shape
    
    A_expanded = A.unsqueeze(-1).unsqueeze(-1)  # 形状: (batch_size, m, n, 1, 1)
    B_expanded = B.unsqueeze(1).unsqueeze(1)      # 形状: (batch_size, 1, 1, p, q)
    
    kron_prod = A_expanded * B_expanded
    result = kron_prod.reshape(batch_size, m * p, n * q)
    
    return result

def N_order_kronecker_expand(x: torch.Tensor, order: int) -> torch.Tensor:
    """
    Args
    ---
        x (Tensor) : 输入张量，形状为 (batch_size, m, n)  
        order (int) : 展开阶数  
    Return
    ---
        Norder_result (Tensor) : 展开后的结果，形状为 (batch_size, m^order, n^order)   
    """
    if order == 1:
        return x
    Norder_result = x
    for i in range(order - 1):
        Norder_result = kronecker_product(Norder_result, x)
    return Norder_result