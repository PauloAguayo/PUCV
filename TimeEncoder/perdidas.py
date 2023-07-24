import torch
import torch.nn.functional as F

# Datos de ejemplo
predicciones = torch.tensor([0.1, 0.05, 0.02, 0.0])
objetivos = torch.tensor([0.15, 0.04, 0.03,0.0001])

# Cálculo del error logarítmico suavizado
error_logaritmico_suavizado = F.smooth_l1_loss(torch.log(predicciones + 1e-6), torch.log(objetivos + 1e-6))
error_logaritmico = F.l1_loss(torch.log(predicciones + 1e-6), torch.log(objetivos + 1e-6))

print("Error logarítmico suavizado:", error_logaritmico_suavizado.item())
print("Error logarítmico:", error_logaritmico.item())
