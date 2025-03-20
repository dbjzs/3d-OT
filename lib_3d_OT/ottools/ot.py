import torch

def sinkhorn(feature1, feature2, pcloud1, pcloud2, epsilon, gamma, max_iter, sim_k=50, dist_k=10):

    feature1 = feature1 / torch.sqrt(torch.sum(feature1 ** 2, -1, keepdim=True) + 1e-8)
    feature2 = feature2 / torch.sqrt(torch.sum(feature2 ** 2, -1, keepdim=True) + 1e-8)
    

    S = torch.bmm(feature1, feature2.transpose(1, 2))

    batch_size, N, M = S.shape
    support = torch.zeros_like(S) 
    
    for b in range(batch_size):
        for n in range(N):
            distances = torch.norm(pcloud1[b, n, :].unsqueeze(0) - pcloud2[b, :, :], p=2, dim=-1)  # [M]

            _, dist_indices = torch.topk(distances, k=dist_k, largest=False)  # [dist_k]

            dist_candidate_indices = dist_indices  # [dist_k]
            candidate_sim_values = S[b, n, dist_candidate_indices]  # [dist_k]
 
            _, sim_indices_from_dist = torch.topk(candidate_sim_values, k=sim_k, largest=True)  # [sim_k]
    
            support[b, n, dist_candidate_indices[sim_indices_from_dist]] = 1.0


    C = 1.0 - S 

    K = torch.exp(-C / epsilon) * support 

    if max_iter == 0:
        return K, S

    power = gamma / (gamma + epsilon)
    a = torch.ones((K.shape[0], K.shape[1], 1), device=feature1.device, dtype=feature1.dtype) / K.shape[1]  # [B, N, 1]
    prob1 = torch.ones((K.shape[0], K.shape[1], 1), device=feature1.device, dtype=feature1.dtype) / K.shape[1]  # [B, N, 1]
    prob2 = torch.ones((K.shape[0], K.shape[2], 1), device=feature2.device, dtype=feature2.dtype) / K.shape[2]  # [B, M, 1]

    for _ in range(max_iter):

        KTa = torch.bmm(K.transpose(1, 2), a)  # [B, M, 1]
        b = torch.pow(prob2 / (KTa + 1e-8), power)  # [B, M, 1]

        Kb = torch.bmm(K, b)  # [B, N, 1]
        a = torch.pow(prob1 / (Kb + 1e-8), power)  # [B, N, 1]

    T = torch.mul(torch.mul(a, K), b.transpose(1, 2))  # [B, N, M]

    return T, S

